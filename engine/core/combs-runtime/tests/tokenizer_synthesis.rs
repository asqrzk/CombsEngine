//! The synthesized-tokenizer parity gate.
//!
//! A bytes-loaded GGUF (the browser's only option) synthesizes its
//! tokenizer from GGUF metadata, while a path-loaded one uses the
//! sibling `tokenizer.json` — so a synthesis bug ships garbage to every
//! browser tab while native stays clean, which is exactly how gemma3
//! shipped `<pad>`-era prompts mangled by a BPE tokenizer synthesized
//! over a SentencePiece vocabulary. This test holds the two sources to
//! encode/decode parity over a battery that exercises the places they
//! can drift: word markers, byte fallback, digits, control tokens.
//!
//! ```text
//! COMBS_TEST_GGUF=$HOME/.cache/combs/models/gemma-3-1b-it-gguf/model.gguf \
//!   cargo test --release -p combs-runtime --test tokenizer_synthesis -- --ignored --nocapture
//! ```

use tokenizers::Tokenizer;

const BATTERY: &[&str] = &[
    "what can you do?",
    "Hello, world! The capital of France is Paris.",
    "  leading and   internal   spaces",
    "line one\nline two\n\nline four",
    "fn main() { println!(\"{:?}\", vec![1, 2, 3]); }",
    "digits 1234567890 and mixed a1b22c333",
    "unicode: naïve café — “quotes” … 中文 émojis 😊🎉",
    "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n",
    "tab\tseparated\tvalues",
    "ends with space ",
];

#[test]
#[ignore = "requires COMBS_TEST_GGUF"]
fn synthesized_tokenizer_matches_the_sibling() {
    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let from_path = combs_formats::open_model_source(&path).expect("path source");
    let bytes = std::fs::read(&path).expect("read model bytes");
    let from_bytes = combs_formats::open_model_source_bytes(bytes).expect("bytes source");

    let sibling = Tokenizer::from_bytes(
        from_path
            .tokenizer()
            .expect("sibling spec")
            .json_bytes()
            .expect("sibling bytes"),
    )
    .expect("sibling tokenizer parses");
    let synthesized = Tokenizer::from_bytes(
        from_bytes
            .tokenizer()
            .expect("synth spec")
            .json_bytes()
            .expect("synth bytes"),
    )
    .expect("synthesized tokenizer parses");

    let mut failures = Vec::new();
    for text in BATTERY {
        let a = sibling.encode(*text, false).expect("sibling encode");
        let b = synthesized.encode(*text, false).expect("synth encode");
        if a.get_ids() != b.get_ids() {
            failures.push(format!(
                "encode drift on {text:?}:\n  sibling {:?}\n  synth   {:?}",
                a.get_ids(),
                b.get_ids()
            ));
            continue;
        }
        let da = sibling.decode(a.get_ids(), false).expect("sibling decode");
        let db = synthesized.decode(b.get_ids(), false).expect("synth decode");
        if da != db {
            failures.push(format!(
                "decode drift on {text:?}: sibling {da:?} vs synth {db:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "synthesized tokenizer drifts from the sibling:\n{}",
        failures.join("\n")
    );
    println!(
        "[tokenizer] synthesized == sibling across {} battery strings",
        BATTERY.len()
    );
}

/// The chat template must actually render — a silent fallback to the
/// token-sniffed wrap discards the checkpoint's own conversation
/// discipline (qwen3's reasoning-stripping included), which is how
/// think-block pollution reached the context. This drives the REAL
/// template from the model file through the real engine wrap.
#[test]
#[ignore = "requires COMBS_TEST_GGUF"]
fn chat_template_renders_and_strips_prior_reasoning() {
    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let source = combs_formats::open_model_source(&path).expect("open model");
    let engine =
        combs_runtime::Engine::load(&source, combs_core::init_device()).expect("load");
    let msgs = vec![
        combs_runtime::ChatMessage {
            role: "user".into(),
            content: "who are you".into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        },
        combs_runtime::ChatMessage {
            role: "assistant".into(),
            content: "<think>\nreasoning here\n</think>\n\nI am an assistant.".into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        },
        combs_runtime::ChatMessage {
            role: "user".into(),
            content: "what can you do".into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        },
    ];
    let prompt = engine.wrap_chat(&msgs);
    println!("[wrap] rendered {} chars:\n{}", prompt.len(), prompt);
    assert!(
        !prompt.contains("reasoning here"),
        "prior-turn reasoning must be stripped from the rendered history"
    );
}
