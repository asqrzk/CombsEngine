//! The segmented-image keystone: a GGUF opened from fixed-size
//! segments must be indistinguishable from the same file opened
//! contiguously — metadata, tokenizer, every dense payload, every
//! packed quant stream, byte for byte. The segment length is chosen
//! far below production (4 MiB vs 1 GiB) so most tensors straddle boundaries and the
//! joining path is exercised everywhere, not just at gigabyte seams.
//!
//! Run:
//! ```sh
//! cargo test -p combs-formats --test gguf_segments -- --ignored --nocapture
//! ```
//! (uses ../../../models/SmolLM2-135M-Instruct-Q8_0.gguf, or COMBS_TEST_GGUF.)

use combs_formats::{GgufSource, ModelSource};

fn model_path() -> String {
    std::env::var("COMBS_TEST_GGUF")
        .unwrap_or_else(|_| "../../../models/SmolLM2-135M-Instruct-Q8_0.gguf".to_string())
}

#[test]
#[ignore = "requires a local GGUF (COMBS_TEST_GGUF)"]
fn segmented_image_is_indistinguishable() {
    let path = model_path();
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipping: {path} not present");
        return;
    };

    // Big enough for the header (the kv block carries the whole vocab),
    // small enough that weight tensors straddle seams constantly.
    const SEG: usize = 4 * 1024 * 1024;
    let segments: Vec<Vec<u8>> = bytes.chunks(SEG).map(<[u8]>::to_vec).collect();
    let whole = GgufSource::from_bytes(bytes).expect("contiguous parse");
    let split = GgufSource::from_segments(segments, SEG).expect("segmented parse");

    assert_eq!(
        format!("{:?}", whole.metadata()),
        format!("{:?}", split.metadata()),
        "metadata must match"
    );
    assert_eq!(
        whole.tokenizer().unwrap().json_bytes().unwrap(),
        split.tokenizer().unwrap().json_bytes().unwrap(),
        "tokenizer bytes must match"
    );

    let mut names = whole.tensor_names();
    names.sort();
    assert!(!names.is_empty());
    let mut straddled = 0usize;
    for name in &names {
        let a = whole.open_tensor(name).expect("whole open");
        let b = split.open_tensor(name).expect("split open");
        assert_eq!(a.shape(), b.shape(), "{name} shape");
        assert_eq!(a.dtype(), b.dtype(), "{name} dtype");
        let av: Vec<f32> = a.load_data().unwrap().to_vec().unwrap();
        let bv: Vec<f32> = b.load_data().unwrap().to_vec().unwrap();
        assert_eq!(av.len(), bv.len(), "{name} len");
        for (i, (x, y)) in av.iter().zip(&bv).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{name}[{i}]");
        }

        let qa = whole.open_tensor_quant(name).expect("whole quant");
        let qb = split.open_tensor_quant(name).expect("split quant");
        match (qa, qb) {
            (None, None) => {}
            (Some(qa), Some(qb)) => {
                assert_eq!(qa.format, qb.format, "{name} quant format");
                assert_eq!(qa.shape, qb.shape, "{name} quant shape");
                assert_eq!(qa.data, qb.data, "{name} packed bytes");
                if matches!(qb.data, std::borrow::Cow::Owned(_)) {
                    straddled += 1;
                }
            }
            (a, b) => panic!("{name}: quant availability differs ({:?} vs {:?})", a.is_some(), b.is_some()),
        }
    }
    println!(
        "[gguf-segments] {} tensors identical; {} packed payloads crossed segment seams",
        names.len(),
        straddled
    );
    assert!(straddled > 0, "4 MiB segments must force straddling — the joining path went untested");
}
