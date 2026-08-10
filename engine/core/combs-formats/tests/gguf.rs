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
    assert!(spec.tokenizer_json.exists());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&spec.tokenizer_json).unwrap()).unwrap();
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
    let tok = tokenizers::Tokenizer::from_file(&spec.tokenizer_json).expect("load tokenizer");
    let enc = tok.encode("123", false).expect("encode");
    assert_eq!(enc.get_ids(), &[1, 2, 3], "digit run must stay split");

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
