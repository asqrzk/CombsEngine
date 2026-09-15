//! The cache scrub spike, before any seam: what a freed buffer holds when the pool hands its page
//! back, why multiplying by zero is not a scrub, and whether one constant write on a tensor's own
//! buffer clears it on each backend the engine runs. Each test initializes the device itself, so
//! each runs in its own process (`--ignored --exact --test-threads=1`).

use burn::backend::wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
use burn::tensor::{DType, Int, Tensor, TensorData};
use cubecl::prelude::*;

use super::{launch, ScrubFill};
use crate::qlinear::{FusedF32, UnfusedF16, UnfusedF32};

const SHAPE: [usize; 4] = [4, 2, 16, 64];

use super::scrub::fill_grid as grid;

fn bytes_of(t: &CubeTensor<WgpuRuntime>) -> usize {
    t.meta.num_elements() * t.dtype.size()
}

fn fill_zero(t: &CubeTensor<WgpuRuntime>) {
    super::scrub::scrub_buffer(t);
    cubecl::future::block_on(t.client.sync()).expect("sync after the fill");
}

fn nonzero_words(bytes: &[u8]) -> usize {
    bytes.chunks_exact(4).filter(|w| w != &[0, 0, 0, 0]).count()
}

fn read(t: &CubeTensor<WgpuRuntime>) -> Vec<u8> {
    let mut back = cubecl::future::block_on(t.client.read_async(vec![t.handle.clone()])).expect("readback");
    back.remove(0).to_vec()
}

/// After the scrub, the tensor is dropped and a buffer of the same size is taken from the pool; this
/// prints how much of it is not zero (the residue the next allocation would see).
fn residue_after_drop(client: &ComputeClient<WgpuRuntime>, bytes: usize) -> usize {
    let h = client.empty(bytes);
    let mut back = cubecl::future::block_on(client.read_async(vec![h])).expect("readback");
    nonzero_words(&back.remove(0))
}

fn pattern_f32() -> Vec<f32> {
    (0..SHAPE.iter().product::<usize>()).map(|i| (i % 997) as f32 + 1.0).collect()
}

#[test]
#[ignore = "gpu: initializes the device itself, so it needs its own process"]
fn a_freed_buffer_comes_back_holding_its_bytes() {
    if crate::skip_no_gpu() {
        return;
    }
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let words: Vec<u32> = (0..(1usize << 18)).map(|i| 0xA5A5_0000 | (i as u32 & 0xFFFF)).collect();
    let bytes = words.len() * 4;
    let h = client.create_from_slice(u32::as_bytes(&words));
    cubecl::future::block_on(client.sync()).expect("sync after upload");
    drop(h);
    let again = client.empty(bytes);
    let mut back = cubecl::future::block_on(client.read_async(vec![again])).expect("readback");
    let back = back.remove(0);
    let same = back
        .chunks_exact(4)
        .zip(&words)
        .filter(|(b, w)| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) == **w)
        .count();
    eprintln!("[scrub-spike] a freed {bytes}-byte buffer, taken again at the same size: {same} of {} words still hold the pattern", words.len());
}

#[test]
#[ignore = "gpu: initializes the device itself, so it needs its own process"]
fn multiplying_by_zero_keeps_the_sign_bit_and_nan() {
    if crate::skip_no_gpu() {
        return;
    }
    let device = WgpuDevice::default();
    let xs = vec![-1.5f32, 2.0, -3.25, 4.0, f32::NAN, -0.5, 6.0, -7.0];
    let negatives = xs.iter().filter(|x| x.is_sign_negative() && !x.is_nan()).count();
    let t: Tensor<UnfusedF32, 1> = Tensor::from_data(TensorData::new(xs, [8]), (&device, DType::F32));
    let got = t.mul_scalar(0.0).into_data().to_vec::<f32>().expect("f32 data");
    let neg_zero = got.iter().filter(|x| x.to_bits() == 0x8000_0000).count();
    let nan = got.iter().filter(|x| x.is_nan()).count();
    eprintln!("[scrub-spike] x * 0: {neg_zero} words read -0.0 (0x80000000) for {negatives} negatives; {nan} NaN kept");
    assert_eq!(neg_zero, negatives, "a multiply by zero keeps each negative value's sign bit");
    assert_eq!(nan, 1, "a multiply by zero keeps NaN");
}

#[test]
#[ignore = "gpu: initializes the device itself, so it needs its own process"]
fn one_constant_write_clears_the_tensors_own_buffer_on_every_backend() {
    if crate::skip_no_gpu() {
        return;
    }
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let f32_t: Tensor<UnfusedF32, 4> = Tensor::from_data(TensorData::new(pattern_f32(), SHAPE), (&device, DType::F32));
    let f16_t: Tensor<UnfusedF16, 4> = Tensor::from_data(TensorData::new(pattern_f32(), SHAPE), (&device, DType::F16));
    let ints: Vec<i32> = (0..SHAPE.iter().product::<usize>()).map(|i| ((i as i32 % 251) - 125) | 1).collect();
    let int_t: Tensor<UnfusedF32, 4, Int> = Tensor::from_data(TensorData::new(ints, SHAPE), (&device, DType::I32));
    let fused_t: Tensor<FusedF32, 4> = Tensor::from_data(TensorData::new(pattern_f32(), SHAPE), (&device, DType::F32));

    // The control: the same bytes dropped without a fill. A zero read after a fill says something only if the pool
    // hands a dropped page of this size back holding its bytes.
    let control: Tensor<UnfusedF32, 4> = Tensor::from_data(TensorData::new(pattern_f32(), SHAPE), (&device, DType::F32));
    let control = control.into_primitive().tensor();
    let control_bytes = bytes_of(&control);
    let _ = read(&control);
    drop(control);
    let control_residue = residue_after_drop(&client, control_bytes);
    eprintln!("[scrub-spike] control (no fill): {control_bytes} bytes dropped · a same-size buffer from the pool afterwards: {control_residue} nonzero");

    let arms: Vec<(&str, CubeTensor<WgpuRuntime>)> = vec![
        ("unfused f32", f32_t.into_primitive().tensor()),
        ("unfused f16", f16_t.into_primitive().tensor()),
        ("unfused i32", int_t.into_primitive()),
        ("fused f32", {
            let ft = fused_t.into_primitive().tensor();
            ft.client.clone().resolve_tensor_float::<UnfusedF32>(ft)
        }),
    ];
    for (name, t) in arms {
        let bytes = bytes_of(&t);
        let before = nonzero_words(&read(&t));
        fill_zero(&t);
        let after = nonzero_words(&read(&t));
        drop(t);
        let residue = residue_after_drop(&client, bytes);
        eprintln!("[scrub-spike] {name}: {bytes} bytes · nonzero words before {before} · after the fill {after} · a same-size buffer from the pool afterwards: {residue} nonzero");
        assert!(before > 0, "{name}: the fixture must hold bytes to clear");
        assert_eq!(after, 0, "{name}: one constant write clears the tensor's own buffer");
    }
}

/// The fill observed as work, not as zeros: at each size one fill is one dispatch, its launch-to-sync wall grows with
/// the bytes, and a fill told to clear nothing leaves every word for almost nothing. Then four phases the runner lines
/// up against the OS's own GPU utilization: idle, a loop of fills told to clear nothing, a loop of real fills over
/// 256 MiB, idle again.
#[test]
#[ignore = "gpu: initializes the device itself and reads the process-wide counters, so it needs its own process"]
fn the_fill_is_gpu_work_in_proportion_to_its_bytes() {
    use std::time::{Duration, Instant};
    if crate::skip_no_gpu() {
        return;
    }
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let pattern = |words: usize| -> Vec<u32> { (0..words).map(|i| 0x5A00_0001 | ((i as u32 & 0xFFFF) << 8)).collect() };

    // Warm: the first launch compiles the pipeline, which is not a fill's cost.
    let warm = client.create_from_slice(u32::as_bytes(&pattern(256)));
    let (count, row) = grid(256);
    launch(&client, ScrubFill, count, vec![warm.clone().binding()], vec![256, row]);
    cubecl::future::block_on(client.sync()).expect("sync after the warm-up");

    for words in [1usize << 18, 1 << 22, 1 << 26] {
        let bytes = words * 4;
        for (arm, clear) in [("told to clear nothing", 0u64), ("the fill", words as u64)] {
            let h = client.create_from_slice(u32::as_bytes(&pattern(words)));
            cubecl::future::block_on(client.sync()).expect("sync after upload");
            let (count, row) = grid(words);
            let (d0, s0) = combs_core::gpu_counters();
            let t0 = Instant::now();
            launch(&client, ScrubFill, count, vec![h.clone().binding()], vec![clear, row]);
            cubecl::future::block_on(client.sync()).expect("sync after the fill");
            let wall = t0.elapsed();
            let (d1, s1) = combs_core::gpu_counters();
            let mut back = cubecl::future::block_on(client.read_async(vec![h])).expect("readback");
            let left = nonzero_words(&back.remove(0));
            eprintln!(
                "[scrub-profile] {bytes} bytes · {arm}: {} dispatches · {} submits · launch to sync {:.3} ms ({:.2} GB/s) · nonzero words left {left} of {words}",
                d1 - d0,
                s1 - s0,
                wall.as_secs_f64() * 1e3,
                bytes as f64 / 1e9 / wall.as_secs_f64()
            );
            assert_eq!(d1 - d0, 1, "{arm}: one fill is one dispatch");
            assert_eq!(left, if clear == 0 { words } else { 0 }, "{arm}: nonzero words left");
        }
    }

    let words = 1usize << 26;
    let h = client.create_from_slice(u32::as_bytes(&pattern(words)));
    cubecl::future::block_on(client.sync()).expect("sync after upload");
    let run_phase = |name: &str, clear: Option<u64>, secs: u64| {
        eprintln!("[scrub-profile] phase {name} begins");
        let t0 = Instant::now();
        let mut launches = 0usize;
        while t0.elapsed() < Duration::from_secs(secs) {
            match clear {
                None => std::thread::sleep(Duration::from_millis(50)),
                Some(c) => {
                    let (count, row) = grid(words);
                    launch(&client, ScrubFill, count, vec![h.clone().binding()], vec![c, row]);
                    cubecl::future::block_on(client.sync()).expect("sync in the loop");
                    launches += 1;
                }
            }
        }
        eprintln!("[scrub-profile] phase {name} ends · {launches} launches");
    };
    run_phase("idle", None, 4);
    run_phase("control-loop", Some(0), 4);
    run_phase("fill-loop", Some(words as u64), 4);
    run_phase("idle-after", None, 4);
}
