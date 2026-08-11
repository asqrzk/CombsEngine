//! Harmony chat-template renders: fixtures in `tests/data/` pair each real
//! checkpoint template (llama-3.2, qwen2.5-coder, gemma-3, smollm2 — pulled
//! from the actual cached models' GGUF metadata / tokenizer_config.json)
//! with reference output rendered by Python `jinja2` under transformers'
//! environment settings (trim_blocks + lstrip_blocks, pinned
//! `strftime_now`). Our minijinja path must match byte-for-byte.

use combs_runtime::ChatTemplate;

#[derive(serde::Deserialize)]
struct Fixture {
    name: String,
    template: String,
    bos_token: String,
    eos_token: String,
    date: String,
    messages: Vec<Message>,
    expected: String,
}

#[derive(serde::Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[test]
fn harmony_renders_match_transformers_reference() {
    let data = include_str!("data/chat_template_harmony.json");
    let fixtures: Vec<Fixture> = serde_json::from_str(data).expect("parse fixtures");
    assert!(fixtures.len() >= 12, "fixture file looks truncated");

    for f in &fixtures {
        // Pin strftime_now to the date the reference was generated with.
        std::env::set_var("COMBS_CHAT_DATE", &f.date);
        let template = ChatTemplate::new(
            f.template.clone(),
            f.bos_token.clone(),
            f.eos_token.clone(),
        );
        let messages: Vec<(String, String)> = f
            .messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        let rendered = template
            .render(&messages)
            .unwrap_or_else(|e| panic!("{}: render failed: {e}", f.name));
        assert_eq!(
            rendered, f.expected,
            "{}: minijinja output diverges from the jinja2 reference",
            f.name
        );
    }
    std::env::remove_var("COMBS_CHAT_DATE");
}
