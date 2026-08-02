//! The spec quick start, verbatim, plus a look at both serializations.

use combs_mesh::{EmojiBuilder, EmojiExporter};

fn main() {
    let emoji = EmojiBuilder::new("my-emoji")
        .description("An emoji that does things")
        .add_todo("task1", "Build the thing")
        .add_image_rgba(64, 64, vec![0u8; 64 * 64 * 4])
        .with_agent_lifecycle()
        .build();

    let binary = EmojiExporter::to_binary(&emoji).expect("to_binary");
    let unicode = EmojiExporter::to_unicode(&emoji).expect("to_unicode");

    println!("emoji:      {}", emoji.name);
    println!("blocks:     {}", emoji.blocks.len());
    println!("binary:     {} bytes", binary.len());
    println!("unicode:    {} chars", unicode.chars().count());

    let back = EmojiExporter::from_binary(&binary).expect("from_binary");
    assert_eq!(emoji, back, "binary round-trip");
    let back = EmojiExporter::from_unicode(&unicode).expect("from_unicode");
    assert_eq!(emoji, back, "unicode round-trip");
    println!("round-trip: ok");
}
