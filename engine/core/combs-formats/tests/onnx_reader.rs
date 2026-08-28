//! ONNX container-reader tests against the generated fixtures
//! (tools/harmony/gen_onnx_fixture.py) plus a truncation matrix over
//! the same bytes.
//!
//! Run:
//! ```sh
//! COMBS_ONNX_FIXTURE_DIR=$HOME/.cache/combs/onnx-fixtures \
//!   cargo test -p combs-formats --test onnx_reader -- --ignored --nocapture
//! ```

use combs_formats::{OnnxData, OnnxDtype, OnnxModel};

fn fixture_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("COMBS_ONNX_FIXTURE_DIR").map(Into::into)
}

fn load(name: &str) -> (Vec<u8>, serde_json::Value) {
    let dir = fixture_dir().expect("COMBS_ONNX_FIXTURE_DIR");
    let buf = std::fs::read(dir.join(name)).expect(name);
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("expected.json")).unwrap())
            .unwrap();
    (buf, expected)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
#[ignore = "requires COMBS_ONNX_FIXTURE_DIR (gen_onnx_fixture.py output)"]
fn plain_initializers_parse_with_inline_ranges() {
    let (buf, expected) = load("plain.onnx");
    let model = OnnxModel::parse(&buf).expect("parse");
    let want = &expected["plain"];

    assert_eq!(model.graph_inputs, vec!["input_ids"]);
    assert_eq!(model.graph_outputs, vec!["logits"]);
    assert_eq!(model.tensors.len(), 3);

    for (name, dtype) in [
        ("model.layers.0.mlp.weight", OnnxDtype::F32),
        ("model.layers.0.attn.weight", OnnxDtype::F16),
        ("model.rope.positions", OnnxDtype::I64),
    ] {
        let t = model.tensors.get(name).unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(t.dtype, dtype, "{name}");
        let dims: Vec<u64> = want[name]["dims"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(t.dims, dims, "{name} dims");
        let OnnxData::Inline { offset, len } = &t.data else {
            panic!("{name} should be inline");
        };
        assert_eq!(
            hex(&buf[*offset..offset + len]),
            want[name]["bytes_hex"].as_str().unwrap(),
            "{name} bytes"
        );
    }
}

#[test]
#[ignore = "requires COMBS_ONNX_FIXTURE_DIR (gen_onnx_fixture.py output)"]
fn external_tensors_carry_offset_and_length() {
    let (buf, expected) = load("external.onnx");
    let model = OnnxModel::parse(&buf).expect("parse");
    let dir = fixture_dir().unwrap();
    let want = &expected["external"];

    // The (offset, length) windows must both match the manifest AND
    // slice the actual sidecar to the original tensor bytes.
    let data = std::fs::read(dir.join("external.onnx.data")).expect("sidecar");
    let plain_want = &expected["plain"];
    for name in [
        "model.layers.0.mlp.weight",
        "model.layers.0.attn.weight",
        "model.rope.positions",
    ] {
        let t = model.tensors.get(name).unwrap_or_else(|| panic!("{name} missing"));
        let OnnxData::External { location, offset, length } = &t.data else {
            panic!("{name} should be external");
        };
        assert_eq!(location, want[name]["location"].as_str().unwrap());
        assert_eq!(*offset, want[name]["offset"].as_u64().unwrap(), "{name} offset");
        assert_eq!(*length, want[name]["length"].as_u64().unwrap(), "{name} length");
        let slice = &data[*offset as usize..(*offset + *length) as usize];
        assert_eq!(
            hex(slice),
            plain_want[name]["bytes_hex"].as_str().unwrap(),
            "{name} sidecar bytes"
        );
    }
}

#[test]
#[ignore = "requires COMBS_ONNX_FIXTURE_DIR (gen_onnx_fixture.py output)"]
fn matmul_nbits_attributes_survive() {
    let (buf, expected) = load("nbits.onnx");
    let model = OnnxModel::parse(&buf).expect("parse");
    let want = &expected["nbits"];

    assert_eq!(model.matmul_nbits.len(), 1);
    let node = &model.matmul_nbits[0];
    assert_eq!(node.k, want["K"].as_u64().unwrap());
    assert_eq!(node.n, want["N"].as_u64().unwrap());
    assert_eq!(node.bits, want["bits"].as_u64().unwrap());
    assert_eq!(node.block_size, want["block_size"].as_u64().unwrap());
    let inputs: Vec<&str> = want["inputs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(node.inputs, inputs);

    let qw = model.tensors.get("qweight").expect("qweight");
    assert_eq!(qw.dtype, OnnxDtype::U8);
    let OnnxData::Inline { offset, len } = &qw.data else {
        panic!("qweight inline");
    };
    assert_eq!(hex(&buf[*offset..offset + len]), want["qweight_bytes_hex"].as_str().unwrap());
    let qs = model.tensors.get("qscales").expect("qscales");
    assert_eq!(qs.dtype, OnnxDtype::F32);
    let OnnxData::Inline { offset, len } = &qs.data else {
        panic!("qscales inline");
    };
    assert_eq!(hex(&buf[*offset..offset + len]), want["qscales_bytes_hex"].as_str().unwrap());
}

#[test]
#[ignore = "requires COMBS_ONNX_FIXTURE_DIR (gen_onnx_fixture.py output)"]
fn truncation_matrix_errs_cleanly() {
    let (buf, _) = load("plain.onnx");
    // Every strict prefix either parses to a model missing pieces or
    // errors — it must never panic. Cut points sweep the whole file.
    for cut in 0..buf.len() {
        let _ = OnnxModel::parse(&buf[..cut]);
    }
    // A cut inside the graph message must be an error, not a silent
    // half-model.
    assert!(OnnxModel::parse(&buf[..buf.len() / 2]).is_err(), "half file must not parse");
    // Corrupt one interior byte at a time near the front (tag bytes) —
    // still never a panic.
    for i in 0..buf.len().min(64) {
        let mut bad = buf.clone();
        bad[i] ^= 0xff;
        let _ = OnnxModel::parse(&bad);
    }
}
