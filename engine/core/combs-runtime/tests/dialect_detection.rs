//! Which dialect a real chat template teaches.
//!
//! The fixtures are the `tokenizer.chat_template` strings lifted out of
//! two GGUF headers — no weights, just the file's own instruction to the
//! model. They are here because the defect this guards against is
//! invisible to a synthetic fixture: BOTH templates contain
//! `<tool_call>`, so a detector that asks the broader dialect first hands
//! the newer family a parser that cannot read it, and the model's
//! perfectly good call comes back as prose.

use combs_runtime::ToolCallStyle;

const QWEN3: &str = include_str!("data/qwen3-chat-template.jinja");
const QWEN38: &str = include_str!("data/qwen38-chat-template.jinja");

#[test]
fn the_fixtures_are_the_templates_the_defect_was_found_in() {
    // If these stop holding, the fixtures were replaced by something
    // that cannot show the bug, and the test below would pass for the
    // wrong reason.
    assert!(QWEN3.contains("<tool_call>"), "qwen3 fixture lost its marker");
    assert!(
        QWEN38.contains("<tool_call>"),
        "qwen3.8 fixture lost its marker"
    );
    assert!(
        !QWEN3.contains("<function="),
        "qwen3 must NOT contain the xml form, or it is not the discriminator"
    );
    assert!(
        QWEN38.contains("<function="),
        "qwen3.8 fixture lost the form that tells it apart"
    );
}

#[test]
fn detection_is_by_specificity() {
    assert_eq!(
        ToolCallStyle::detect(Some(QWEN38)),
        ToolCallStyle::XmlFunction,
        "qwen3.8 writes <function=…> and must not be handed the json parser"
    );
    assert_eq!(
        ToolCallStyle::detect(Some(QWEN3)),
        ToolCallStyle::Hermes,
        "qwen3 writes json inside <tool_call> and must keep the parser it had"
    );
}

#[test]
fn both_families_take_the_same_thinking_switch_but_open_it_differently() {
    let qwen3 = combs_runtime::dialect::for_style(ToolCallStyle::Hermes).thinking();
    let qwen38 = combs_runtime::dialect::for_style(ToolCallStyle::XmlFunction).thinking();

    // Read from the fixtures: both honour `enable_thinking`, and only
    // qwen3.8 has the else-branch that opens the thought in the PROMPT.
    assert!(QWEN3.contains("enable_thinking"));
    assert!(QWEN38.contains("enable_thinking"));
    assert_eq!(qwen3.off, combs_runtime::dialect::Off::Kwarg("enable_thinking"));
    assert_eq!(qwen38.off, combs_runtime::dialect::Off::Kwarg("enable_thinking"));

    assert!(
        !qwen3.open_in_prompt,
        "qwen3 emits both tags itself"
    );
    assert!(
        qwen38.open_in_prompt,
        "qwen3.8 opens the thought in the prompt, so only the close is generated"
    );

    // Neither template mentions the soft switch; it is training, not
    // contract, and a client that relies on it should know that.
    assert!(!QWEN3.contains("/no_think"));
    assert!(!QWEN38.contains("/no_think"));
}
