//! Synthetic-GGUF round-trip test: builds a minimal GGUF v3 file in a temp
//! dir and verifies header parsing, metadata mapping, name mapping and
//! Q8_0 dequantization through the `ModelSource` trait.

use std::io::Write;

use combs_formats::{GgufSource, ModelSource};

/// Builds a tiny llama GGUF: 1 layer, hidden 64, 1 head, vocab 8, one
/// Q8_0 tensor (token_embd) + one F32 tensor (output_norm).
fn write_test_gguf(path: &std::path::Path) {
    let mut out: Vec<u8> = Vec::new();
    let w32 = |v: u32, out: &mut Vec<u8>| out.extend_from_slice(&v.to_le_bytes());
    let w64 = |v: u64, out: &mut Vec<u8>| out.extend_from_slice(&v.to_le_bytes());
    let wstr = |s: &str, out: &mut Vec<u8>| {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    };

    out.extend_from_slice(b"GGUF");
    w32(3, &mut out); // version
    w64(2, &mut out); // tensor count
    w64(10, &mut out); // metadata kv count

    // metadata
    wstr("general.architecture", &mut out);
    w32(8, &mut out); // string
    wstr("llama", &mut out);
    for (key, val) in [
        ("llama.embedding_length", 64u32),
        ("llama.attention.head_count", 1),
        ("llama.attention.head_count_kv", 1),
        ("llama.block_count", 1),
        ("llama.context_length", 128),
        ("llama.feed_forward_length", 128),
    ] {
        wstr(key, &mut out);
        w32(4, &mut out); // u32
        w32(val, &mut out);
    }
    wstr("tokenizer.ggml.eos_token_id", &mut out);
    w32(4, &mut out);
    w32(7, &mut out);
    wstr("tokenizer.ggml.tokens", &mut out);
    w32(9, &mut out); // array
    w32(8, &mut out); // of strings
    w64(8, &mut out); // len
    for tok in ["<s>", "a", "b", "c", "d", "e", "f", "</s>"] {
        wstr(tok, &mut out);
    }
    wstr("tokenizer.ggml.merges", &mut out);
    w32(9, &mut out);
    w32(8, &mut out);
    w64(0, &mut out);

    // tensor infos: token_embd Q8_0 [64, 8] (ggml dims), output_norm F32 [64]
    wstr("token_embd.weight", &mut out);
    w32(2, &mut out); // 2 dims
    w64(64, &mut out); // ggml: in-dim first
    w64(8, &mut out);
    w32(8, &mut out); // Q8_0
    w64(0, &mut out); // offset

    wstr("output_norm.weight", &mut out);
    w32(1, &mut out);
    w64(64, &mut out);
    w32(0, &mut out); // F32
    let q8_bytes = (64 * 8 / 32) * 34;
    w64(q8_bytes as u64, &mut out); // offset

    // align data to 32
    let pad = (32 - (out.len() % 32)) % 32;
    out.extend(std::iter::repeat(0u8).take(pad));

    // Q8_0 data: 16 blocks, scale=0.5 (f16), values alternating 2/-2
    for _ in 0..16 {
        out.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        for j in 0..32 {
            out.push(if j % 2 == 0 { 2u8 } else { 254u8 }); // 254 = -2 as u8
        }
    }
    // F32 output_norm: all 1.0
    for _ in 0..64 {
        out.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&out).unwrap();
}

#[test]
fn gguf_round_trip() {
    let dir = std::env::temp_dir().join(format!("combs-gguf-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.gguf");
    write_test_gguf(&path);

    let source = GgufSource::load(&path).expect("parse");
    let md = source.metadata();
    assert_eq!(md.architecture, "llama");
    assert_eq!(md.hidden_size, 64);
    assert_eq!(md.num_hidden_layers, 1);
    assert_eq!(md.num_attention_heads, 1);
    assert_eq!(md.max_position_embeddings, 128);
    assert_eq!(md.eos_token_ids, vec![7]);
    // tied embeddings: no output.weight in the file
    assert!(md.tie_word_embeddings);

    let names = source.tensor_names();
    assert!(names.contains(&"model.embed_tokens.weight".to_string()));
    assert!(names.contains(&"model.norm.weight".to_string()));

    // Q8_0 dequant: values ±1.0 (2 * 0.5), HF shape = dims reversed.
    let reader = source.open_tensor("model.embed_tokens.weight").expect("tensor");
    assert_eq!(reader.shape(), &[8, 64]);
    let data = reader.load_data().expect("load");
    let values = data.to_vec::<f32>().expect("f32");
    assert_eq!(values.len(), 512);
    assert!((values[0] - 1.0).abs() < 1e-6);
    assert!((values[1] + 1.0).abs() < 1e-6);

    let norm = source.open_tensor("model.norm.weight").expect("norm");
    let values = norm.load_data().unwrap().to_vec::<f32>().unwrap();
    assert!(values.iter().all(|v| (*v - 1.0).abs() < 1e-6));

    // tokenizer.json synthesized from GGUF metadata
    let spec = source.tokenizer().expect("tokenizer");
    let json: serde_json::Value =
        serde_json::from_slice(&spec.json_bytes().unwrap()).unwrap();
    assert_eq!(json["model"]["vocab"]["a"], 1);
    // No add_bos_token key in this file -> unspecified.
    assert_eq!(spec.add_bos, None);

    std::fs::remove_dir_all(&dir).ok();
}

/// Builds a tiny qwen2 GGUF: same two tensors as the llama fixture, plus
/// `add_bos_token=false`, `pre="qwen2"`, and a digit vocab with merges that
/// would fuse "123" into one token if the default (`\p{N}{1,3}`) split regex
/// were used instead of qwen's single-digit split.
fn write_qwen_test_gguf(path: &std::path::Path) {
    let mut out: Vec<u8> = Vec::new();
    let w32 = |v: u32, out: &mut Vec<u8>| out.extend_from_slice(&v.to_le_bytes());
    let w64 = |v: u64, out: &mut Vec<u8>| out.extend_from_slice(&v.to_le_bytes());
    let wstr = |s: &str, out: &mut Vec<u8>| {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    };

    out.extend_from_slice(b"GGUF");
    w32(3, &mut out); // version
    w64(2, &mut out); // tensor count
    w64(13, &mut out); // metadata kv count

    wstr("general.architecture", &mut out);
    w32(8, &mut out); // string
    wstr("qwen2", &mut out);
    for (key, val) in [
        ("qwen2.embedding_length", 64u32),
        ("qwen2.attention.head_count", 1),
        ("qwen2.attention.head_count_kv", 1),
        ("qwen2.block_count", 1),
        ("qwen2.context_length", 128),
        ("qwen2.feed_forward_length", 128),
    ] {
        wstr(key, &mut out);
        w32(4, &mut out); // u32
        w32(val, &mut out);
    }
    wstr("tokenizer.ggml.eos_token_id", &mut out);
    w32(4, &mut out);
    w32(7, &mut out);
    wstr("tokenizer.ggml.bos_token_id", &mut out);
    w32(4, &mut out);
    w32(0, &mut out);
    wstr("tokenizer.ggml.add_bos_token", &mut out);
    w32(7, &mut out); // bool
    out.push(0); // false
    wstr("tokenizer.ggml.pre", &mut out);
    w32(8, &mut out); // string
    wstr("qwen2", &mut out);
    wstr("tokenizer.ggml.tokens", &mut out);
    w32(9, &mut out); // array
    w32(8, &mut out); // of strings
    w64(8, &mut out);
    for tok in ["<s>", "1", "2", "3", "12", "123", "x", "</s>"] {
        wstr(tok, &mut out);
    }
    wstr("tokenizer.ggml.merges", &mut out);
    w32(9, &mut out);
    w32(8, &mut out);
    w64(2, &mut out);
    for merge in ["1 2", "12 3"] {
        wstr(merge, &mut out);
    }

    // tensor infos: token_embd Q8_0 [64, 8] (ggml dims), output_norm F32 [64]
    wstr("token_embd.weight", &mut out);
    w32(2, &mut out);
    w64(64, &mut out);
    w64(8, &mut out);
    w32(8, &mut out); // Q8_0
    w64(0, &mut out);

    wstr("output_norm.weight", &mut out);
    w32(1, &mut out);
    w64(64, &mut out);
    w32(0, &mut out); // F32
    let q8_bytes = (64 * 8 / 32) * 34;
    w64(q8_bytes as u64, &mut out);

    let pad = (32 - (out.len() % 32)) % 32;
    out.extend(std::iter::repeat(0u8).take(pad));

    for _ in 0..16 {
        out.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        for j in 0..32 {
            out.push(if j % 2 == 0 { 2u8 } else { 254u8 });
        }
    }
    for _ in 0..64 {
        out.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&out).unwrap();
}

#[test]
fn qwen_gguf_add_bos_and_single_digit_pretokenizer() {
    let dir = std::env::temp_dir().join(format!("combs-gguf-qwen-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny-qwen.gguf");
    write_qwen_test_gguf(&path);

    // Poison the synthesis cache with the old GPT-2 regex (what earlier
    // engine versions wrote for every family): loading must regenerate it.
    let cached = path.with_extension("tokenizer.json");
    std::fs::write(
        &cached,
        r#"{"version":"1.0","added_tokens":[],"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false}]},"model":{"type":"BPE","vocab":{},"merges":[]}}"#,
    )
    .unwrap();

    let source = GgufSource::load(&path).expect("parse");
    assert_eq!(source.metadata().architecture, "qwen2");

    let spec = source.tokenizer().expect("tokenizer");
    assert_eq!(spec.add_bos, Some(false));

    // The qwen2 pre family splits digit runs into single digits, so "123"
    // must encode as three tokens even though the "1 2"/"12 3" merges could
    // fuse it under the default split regex.
    let tok = tokenizers::Tokenizer::from_bytes(spec.json_bytes().unwrap())
        .expect("load tokenizer");
    let enc = tok.encode("123", false).expect("encode");
    assert_eq!(enc.get_ids(), &[1, 2, 3], "digit run must stay split");

    std::fs::remove_dir_all(&dir).ok();
}

/// Builds a tiny phi3 GGUF: fused `attn_qkv` (hidden 4, 2 heads / 1 kv
/// head → rows [q0 q1 q2 q3 | k0 k1 | v0 v1]) and fused `ffn_up`
/// (intermediate 3 → rows [g0 g1 g2 | u0 u1 u2]), each row filled with its
/// global row index so slices are recognizable. Plus phi3's sliding-window
/// key and an `<|end|>` control token beyond the declared eos.
fn write_phi3_test_gguf(path: &std::path::Path) {
    let mut out: Vec<u8> = Vec::new();
    let w32 = |v: u32, out: &mut Vec<u8>| out.extend_from_slice(&v.to_le_bytes());
    let w64 = |v: u64, out: &mut Vec<u8>| out.extend_from_slice(&v.to_le_bytes());
    let wstr = |s: &str, out: &mut Vec<u8>| {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    };

    out.extend_from_slice(b"GGUF");
    w32(3, &mut out);
    w64(2, &mut out); // tensors: attn_qkv + ffn_up
    w64(11, &mut out); // kv count

    wstr("general.architecture", &mut out);
    w32(8, &mut out);
    wstr("phi3", &mut out);
    for (key, val) in [
        ("phi3.embedding_length", 4u32),
        ("phi3.attention.head_count", 2),
        ("phi3.attention.head_count_kv", 1),
        ("phi3.block_count", 1),
        ("phi3.context_length", 64),
        ("phi3.feed_forward_length", 3),
        ("phi3.attention.sliding_window", 2047),
    ] {
        wstr(key, &mut out);
        w32(4, &mut out);
        w32(val, &mut out);
    }
    wstr("tokenizer.ggml.eos_token_id", &mut out);
    w32(4, &mut out);
    w32(1, &mut out);
    wstr("tokenizer.ggml.tokens", &mut out);
    w32(9, &mut out);
    w32(8, &mut out);
    w64(4, &mut out);
    for tok in ["<s>", "<|endoftext|>", "<|end|>", "a"] {
        wstr(tok, &mut out);
    }
    wstr("tokenizer.ggml.token_type", &mut out);
    w32(9, &mut out);
    w32(5, &mut out); // i32 array
    w64(4, &mut out);
    for ty in [3i32, 3, 3, 1] {
        out.extend_from_slice(&ty.to_le_bytes());
    }

    // tensor infos (ggml dims: in-dim first): attn_qkv [4, 8], ffn_up [4, 6]
    wstr("blk.0.attn_qkv.weight", &mut out);
    w32(2, &mut out);
    w64(4, &mut out);
    w64(8, &mut out);
    w32(0, &mut out); // F32
    w64(0, &mut out);
    wstr("blk.0.ffn_up.weight", &mut out);
    w32(2, &mut out);
    w64(4, &mut out);
    w64(6, &mut out);
    w32(0, &mut out);
    w64(4 * 8 * 4, &mut out); // after attn_qkv (128 B, 32-aligned)

    let pad = (32 - (out.len() % 32)) % 32;
    out.extend(std::iter::repeat(0u8).take(pad));

    // attn_qkv rows 0..8, ffn_up rows 0..6, each row constant = row index.
    for row in 0..8 {
        for _ in 0..4 {
            out.extend_from_slice(&(row as f32).to_le_bytes());
        }
    }
    for row in 0..6 {
        for _ in 0..4 {
            out.extend_from_slice(&(row as f32).to_le_bytes());
        }
    }
    std::fs::write(path, &out).unwrap();
}

#[test]
fn phi3_fused_tensors_serve_split_names() {
    let dir = std::env::temp_dir().join(format!("combs-gguf-phi3-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("phi3.gguf");
    write_phi3_test_gguf(&path);

    let source = GgufSource::load(&path).expect("parse");
    let md = source.metadata();
    assert_eq!(md.architecture, "phi3");
    // Sliding-window key surfaces in the attention pattern.
    assert_eq!(md.attention_pattern.sliding_window, Some(2047));
    // `<|end|>` (id 2) joins the declared eos (id 1) via the EOG scan.
    assert_eq!(md.eos_token_ids, vec![1, 2]);

    // Fused attn_qkv [8,4] serves q/k/v as row ranges [0..4 | 4..6 | 6..8].
    let expect = [
        ("model.layers.0.self_attn.q_proj.weight", vec![4usize, 4], 0.0f32),
        ("model.layers.0.self_attn.k_proj.weight", vec![2, 4], 4.0),
        ("model.layers.0.self_attn.v_proj.weight", vec![2, 4], 6.0),
        ("model.layers.0.mlp.gate_proj.weight", vec![3, 4], 0.0),
        ("model.layers.0.mlp.up_proj.weight", vec![3, 4], 3.0),
    ];
    for (name, shape, first_row) in expect {
        let reader = source.open_tensor(name).expect(name);
        assert_eq!(reader.shape(), &shape[..], "{name} shape");
        let values = reader.load_data().unwrap().to_vec::<f32>().unwrap();
        for (i, v) in values.iter().enumerate() {
            let row = i / shape[1];
            assert_eq!(
                *v,
                first_row + row as f32,
                "{name} row {row} must come from fused row {}",
                first_row + row as f32
            );
        }
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// Builds a tiny gemma3 GGUF: the family's norm names (`ffn_norm` meaning
/// PRE-feedforward — llama's same-named tensor is the pre-MLP post-attn
/// norm — plus sandwich + qk norms), an explicit `attention.key_length`
/// decoupled from hidden/heads, a sliding-window key, and an
/// `<end_of_turn>` control token beyond the declared eos. Each norm is
/// filled with a distinct constant so the mapping is provable.
fn write_gemma3_test_gguf(path: &std::path::Path) {
    let mut out: Vec<u8> = Vec::new();
    let w32 = |v: u32, out: &mut Vec<u8>| out.extend_from_slice(&v.to_le_bytes());
    let w64 = |v: u64, out: &mut Vec<u8>| out.extend_from_slice(&v.to_le_bytes());
    let wstr = |s: &str, out: &mut Vec<u8>| {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    };

    out.extend_from_slice(b"GGUF");
    w32(3, &mut out);
    w64(6, &mut out); // tensors: 6 norms
    w64(12, &mut out); // kv count

    wstr("general.architecture", &mut out);
    w32(8, &mut out);
    wstr("gemma3", &mut out);
    for (key, val) in [
        ("gemma3.embedding_length", 8u32),
        ("gemma3.attention.head_count", 2),
        ("gemma3.attention.head_count_kv", 1),
        ("gemma3.attention.key_length", 6), // ≠ hidden/heads = 4
        ("gemma3.block_count", 1),
        ("gemma3.context_length", 64),
        ("gemma3.feed_forward_length", 16),
        ("gemma3.attention.sliding_window", 512),
    ] {
        wstr(key, &mut out);
        w32(4, &mut out);
        w32(val, &mut out);
    }
    wstr("tokenizer.ggml.eos_token_id", &mut out);
    w32(4, &mut out);
    w32(1, &mut out);
    wstr("tokenizer.ggml.tokens", &mut out);
    w32(9, &mut out);
    w32(8, &mut out);
    w64(4, &mut out);
    for tok in ["<bos>", "<eos>", "<end_of_turn>", "a"] {
        wstr(tok, &mut out);
    }
    wstr("tokenizer.ggml.token_type", &mut out);
    w32(9, &mut out);
    w32(5, &mut out);
    w64(4, &mut out);
    for ty in [3i32, 3, 3, 1] {
        out.extend_from_slice(&ty.to_le_bytes());
    }

    // Six F32 norm tensors, [8] each, row value = index in this list.
    let norms = [
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.post_attention_norm.weight",
        "blk.0.post_ffw_norm.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
    ];
    for (i, name) in norms.iter().enumerate() {
        wstr(name, &mut out);
        w32(1, &mut out);
        w64(8, &mut out);
        w32(0, &mut out); // F32
        w64((i * 8 * 4) as u64, &mut out);
    }

    let pad = (32 - (out.len() % 32)) % 32;
    out.extend(std::iter::repeat(0u8).take(pad));
    for i in 0..norms.len() {
        for _ in 0..8 {
            out.extend_from_slice(&(i as f32).to_le_bytes());
        }
    }
    std::fs::write(path, &out).unwrap();
}

#[test]
fn gemma3_norm_names_and_metadata_map() {
    let dir = std::env::temp_dir().join(format!("combs-gguf-gemma3-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("gemma3.gguf");
    write_gemma3_test_gguf(&path);

    let source = GgufSource::load(&path).expect("parse");
    let md = source.metadata();
    assert_eq!(md.architecture, "gemma3");
    // Explicit key_length wins over hidden/heads (6 vs 4).
    assert_eq!(md.head_dim, 6);
    assert_eq!(md.attention_pattern.sliding_window, Some(512));
    // Defaults mirror llama.cpp's gemma3 hardcoding.
    assert_eq!(md.attention_pattern.pattern, 6);
    assert_eq!(md.attention_pattern.rope_local_theta, 10_000.0);
    // `<end_of_turn>` (id 2) joins the declared eos (id 1).
    assert_eq!(md.eos_token_ids, vec![1, 2]);

    // The gemma3 norm-name arms: same ggml `ffn_norm` maps to a DIFFERENT
    // HF tensor than llama's, and the sandwich/qk norms resolve. llama.cpp
    // bakes `(1+w)` into gemma norm weights, so the adapter serves each
    // fill value MINUS 1 (back to HF semantics).
    let expect = [
        ("model.layers.0.input_layernorm.weight", 0.0f32),
        ("model.layers.0.pre_feedforward_layernorm.weight", 1.0),
        ("model.layers.0.post_attention_layernorm.weight", 2.0),
        ("model.layers.0.post_feedforward_layernorm.weight", 3.0),
        ("model.layers.0.self_attn.q_norm.weight", 4.0),
        ("model.layers.0.self_attn.k_norm.weight", 5.0),
    ];
    for (name, fill) in expect {
        let reader = source.open_tensor(name).expect(name);
        let values = reader.load_data().unwrap().to_vec::<f32>().unwrap();
        let want = fill - 1.0;
        assert!(
            values.iter().all(|v| *v == want),
            "{name} must serve fill {fill} minus the baked gemma +1"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn split_shard_gguf_is_rejected() {
    let dir = std::env::temp_dir().join(format!("combs-gguf-split-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shard.gguf");

    // Minimal header: no tensors, just llama.cpp's u16 split markers.
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    out.extend_from_slice(&2u64.to_le_bytes()); // kv count
    let wstr = |s: &str, out: &mut Vec<u8>| {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    };
    wstr("split.no", &mut out);
    out.extend_from_slice(&2u32.to_le_bytes()); // u16
    out.extend_from_slice(&0u16.to_le_bytes());
    wstr("split.count", &mut out);
    out.extend_from_slice(&2u32.to_le_bytes()); // u16
    out.extend_from_slice(&2u16.to_le_bytes());
    std::fs::write(&path, &out).unwrap();

    let err = GgufSource::load(&path).err().expect("must refuse a split shard");
    let msg = err.to_string();
    assert!(msg.contains("shard 1 of 2"), "unexpected error: {msg}");

    std::fs::remove_dir_all(&dir).ok();
}

/// The in-memory door and the mapped door must open onto the same model.
///
/// Not "both load": every observable — metadata, the tensor name set, the
/// dequantized bytes of each tensor, the packed quant bytes, and the
/// tokenizer JSON — must agree. A browser that gets a *slightly* different
/// model than the CLI is worse than one that gets none, because the
/// difference shows up as bad output rather than as an error.
#[test]
fn from_bytes_matches_load() {
    let dir = std::env::temp_dir().join(format!("combs-gguf-bytes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.gguf");
    write_test_gguf(&path);

    let mapped = GgufSource::load(&path).expect("mapped parse");
    let bytes = GgufSource::from_bytes(std::fs::read(&path).unwrap()).expect("bytes parse");

    assert_eq!(
        format!("{:?}", mapped.metadata()),
        format!("{:?}", bytes.metadata()),
        "metadata must not depend on how the bytes arrived"
    );

    let mut a = mapped.tensor_names();
    let mut b = bytes.tensor_names();
    a.sort();
    b.sort();
    assert_eq!(a, b, "tensor name sets differ");

    for name in &a {
        let ra = mapped.open_tensor(name).expect("mapped tensor");
        let rb = bytes.open_tensor(name).expect("bytes tensor");
        assert_eq!(ra.shape(), rb.shape(), "{name}: shape");
        assert_eq!(ra.dtype(), rb.dtype(), "{name}: dtype");
        assert_eq!(ra.raw_bytes(), rb.raw_bytes(), "{name}: bytes");

        let qa = mapped.open_tensor_quant(name).expect("mapped quant");
        let qb = bytes.open_tensor_quant(name).expect("bytes quant");
        match (qa, qb) {
            (Some(qa), Some(qb)) => {
                assert_eq!(qa.format, qb.format, "{name}: quant format");
                assert_eq!(qa.shape, qb.shape, "{name}: quant shape");
                assert_eq!(qa.data.as_ref(), qb.data.as_ref(), "{name}: packed bytes");
            }
            (None, None) => {}
            _ => panic!("{name}: one source offered packed bytes and the other did not"),
        }
    }

    let sa = mapped.tokenizer().expect("mapped tokenizer");
    let sb = bytes.tokenizer().expect("bytes tokenizer");
    assert_eq!(
        sa.json_bytes().unwrap(),
        sb.json_bytes().unwrap(),
        "synthesized tokenizer JSON differs between the cached file and memory"
    );
    assert_eq!(sa.added_tokens, sb.added_tokens);
    assert_eq!(sa.add_bos, sb.add_bos);
    // The mapped source cached its synthesis next to the model; the
    // in-memory one has no file, and says so rather than inventing a path.
    assert!(sa.json_path().is_some(), "mapped tokenizer should be path-backed");
    assert!(sb.json_path().is_none(), "in-memory tokenizer has no path");

    // Truncation is a wrong model, not a smaller one: GGUF offsets are
    // absolute, so a partial buffer must be refused, not half-read.
    let whole = std::fs::read(&path).unwrap();
    let truncated = whole[..whole.len() / 2].to_vec();
    let short = GgufSource::from_bytes(truncated);
    if let Ok(src) = short {
        assert!(
            src.open_tensor("model.embed_tokens.weight").is_err(),
            "a truncated image must not serve tensor bytes"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}
