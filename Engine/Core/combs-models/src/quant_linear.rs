//! Quantized linear layer: packed 4-bit weights + per-group scales,
//! dequantized on-device before the matmul.
//!
//! The packed weight stays in VRAM in its compact form between forward
//! passes (a 4x footprint reduction vs f32 for the weight); the current
//! implementation dequantizes to f32 and runs the standard matmul (see
//! `combs_core::quant` docs — a fused dequant-matmul kernel is future
//! work). Format adapters (GGUF in Phase 5) construct these layers via
//! [`QuantizedLinear::new`].

use burn::tensor::{Int, Tensor, backend::Backend};

use combs_core::quant::{DEFAULT_Q4_GROUP_SIZE, dequantize_q4};

use crate::matmul::safe_matmul;
use crate::{ModelError, Result};

/// `y = x @ W^T (+ b)` where `W` is stored group-quantized to 4 bits.
pub struct QuantizedLinear<B: Backend> {
    /// `[out_features, in_features / 2]` packed nibbles (GGUF q4_0 order).
    packed: Tensor<B, 2, Int>,
    /// `[out_features, in_features / group_size]` per-block scales.
    scales: Tensor<B, 2>,
    /// Values per quantization block (GGUF q4_0: 32).
    group_size: usize,
    /// Optional `[out_features]` bias.
    bias: Option<Tensor<B, 1>>,
    in_features: usize,
    out_features: usize,
}

impl<B: Backend> QuantizedLinear<B> {
    /// Builds a layer from on-device packed parts, validating shapes.
    pub fn new(
        packed: Tensor<B, 2, Int>,
        scales: Tensor<B, 2>,
        group_size: usize,
        bias: Option<Tensor<B, 1>>,
    ) -> Result<Self> {
        let [out_features, packed_cols] = packed.dims();
        if group_size == 0 || group_size % 2 != 0 {
            return Err(ModelError::BadShape {
                tensor: "quantized_weight".into(),
                expected: vec![DEFAULT_Q4_GROUP_SIZE],
                got: vec![group_size],
            });
        }
        if packed_cols % (group_size / 2) != 0 {
            return Err(ModelError::BadShape {
                tensor: "quantized_weight".into(),
                expected: vec![group_size / 2],
                got: vec![packed_cols],
            });
        }
        let in_features = packed_cols * 2;
        let [s_rows, s_cols] = scales.dims();
        if s_rows != out_features || s_cols != in_features / group_size {
            return Err(ModelError::BadShape {
                tensor: "quantized_scales".into(),
                expected: vec![out_features, in_features / group_size],
                got: vec![s_rows, s_cols],
            });
        }
        if let Some(b) = &bias {
            if b.dims()[0] != out_features {
                return Err(ModelError::BadShape {
                    tensor: "quantized_bias".into(),
                    expected: vec![out_features],
                    got: vec![b.dims()[0]],
                });
            }
        }
        Ok(QuantizedLinear {
            packed,
            scales,
            group_size,
            bias,
            in_features,
            out_features,
        })
    }

    /// Input feature count (dequantized `in_features`).
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output feature count.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Dequantizes the weight to f32 `[out_features, in_features]`.
    pub fn weight(&self) -> Tensor<B, 2> {
        dequantize_q4(self.packed.clone(), self.scales.clone(), self.group_size)
    }

    /// `y = x @ dequant(W)^T (+ b)` for `x: [batch, seq, in_features]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let w = self.weight();
        let out = safe_matmul(x, w.transpose().unsqueeze_dim::<3>(0));
        match &self.bias {
            Some(b) => {
                let [batch, seq, dim] = out.dims();
                out + b.clone().reshape([1, 1, dim]).expand([batch, seq, dim])
            }
            None => out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type B = burn::backend::NdArray<f32>;

    /// Packs f32 weights into GGUF q4_0 form (per-block abs-max scale).
    fn pack_q4(w: &[f32], rows: usize, cols: usize) -> (Vec<i32>, Vec<f32>) {
        let mut packed = vec![0i32; rows * cols / 2];
        let mut scales = vec![0f32; rows * (cols / 32)];
        for r in 0..rows {
            for g in 0..cols / 32 {
                let block = &w[r * cols + g * 32..r * cols + g * 32 + 32];
                let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
                let scale = amax / 7.0;
                scales[r * (cols / 32) + g] = scale;
                for j in 0..16 {
                    let lo = ((block[j] / scale).round() as i32 + 8).clamp(0, 15) as u32;
                    let hi = ((block[j + 16] / scale).round() as i32 + 8).clamp(0, 15) as u32;
                    packed[r * cols / 2 + g * 16 + j] = ((hi << 4) | lo) as i32;
                }
            }
        }
        (packed, scales)
    }

    #[test]
    fn forward_matches_dequantized_dense_matmul() {
        let device = Default::default();
        let (rows, cols) = (4, 64); // out=4, in=64
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 13 % 17) as f32 - 8.0) / 4.0)
            .collect();
        let (packed, scales) = pack_q4(&w, rows, cols);

        let layer = QuantizedLinear::<B>::new(
            Tensor::from_data(TensorData::new(packed, [rows, cols / 2]), &device),
            Tensor::from_data(TensorData::new(scales, [rows, cols / 32]), &device),
            DEFAULT_Q4_GROUP_SIZE,
            None,
        )
        .unwrap();
        assert_eq!(layer.in_features(), cols);
        assert_eq!(layer.out_features(), rows);

        let x_data: Vec<f32> = (0..cols).map(|i| (i as f32) / 16.0).collect();
        let x = Tensor::<B, 3>::from_data(TensorData::new(x_data.clone(), [1, 1, cols]), &device);

        let got = layer.forward(x.clone());
        let dense_w = layer.weight();
        let expect = crate::matmul::safe_matmul(
            x,
            dense_w.transpose().unsqueeze_dim::<3>(0),
        );
        let got: Vec<f32> = got.into_data().to_vec().unwrap();
        let expect: Vec<f32> = expect.into_data().to_vec().unwrap();
        for (g, e) in got.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-5);
        }

        // And the quantized weight must be close to the original (q4 error).
        let deq: Vec<f32> = layer.weight().into_data().to_vec().unwrap();
        let max_err = deq
            .iter()
            .zip(w.iter())
            .fold(0f32, |m, (d, o)| m.max((d - o).abs()));
        assert!(max_err < 0.5, "q4 reconstruction error too large: {max_err}");
    }
}
