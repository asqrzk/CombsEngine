//! Round-trip tests: binary and unicode encodings preserve every block.

use combs_mesh::*;

fn full_emoji() -> Emoji {
    EmojiBuilder::new("roundtrip-emoji")
        .description("exercises every block type")
        .add_todo("task1", "Build the thing")
        .add_image_rgba(8, 8, (0..8 * 8 * 4).map(|i| (i % 251) as u8).collect())
        .with_agent_lifecycle()
        .add_block(Block::Fnc(FunctionBlock {
            definitions: vec![FunctionDef {
                name: "greet".into(),
                kind: FunctionKind::Custom("social".into()),
                params: vec!["who".into()],
                body: "return hello(who)".into(),
            }],
        }))
        .add_block(Block::Api(ApiBlock {
            endpoints: vec![ApiEndpoint {
                name: "status".into(),
                method: "GET".into(),
                path: "/status".into(),
                description: "health check".into(),
            }],
        }))
        .add_block(Block::Chr(CharacterBlock {
            traits: vec![("brave".into(), 0.8), ("curious".into(), 0.5)],
            backstory: "born in a test suite".into(),
        }))
        .add_block(Block::Emo(EmotionBlock {
            states: vec![EmotionState {
                name: "joy".into(),
                intensity: 0.9,
            }],
        }))
        .add_block(Block::Orc(OrchestrationBlock {
            directives: vec![
                OrchestrationDirective {
                    kind: DirectiveKind::Todo,
                    key: "next".into(),
                    value: "task1".into(),
                },
                OrchestrationDirective {
                    kind: DirectiveKind::Note,
                    key: "hint".into(),
                    value: "keep it simple".into(),
                },
            ],
        }))
        .build()
}

#[test]
fn binary_roundtrip_all_blocks() {
    let emoji = full_emoji();
    let bytes = EmojiExporter::to_binary(&emoji).expect("to_binary");
    assert_eq!(&bytes[0..4], b"CMSE");
    let back = EmojiExporter::from_binary(&bytes).expect("from_binary");
    assert_eq!(emoji, back);
}

#[test]
fn unicode_roundtrip_all_blocks() {
    let emoji = full_emoji();
    let encoded = EmojiExporter::to_unicode(&emoji).expect("to_unicode");
    // Every char must be one of our marker ranges — valid scalar values
    // (Rust `String` cannot hold surrogates by construction).
    assert!(encoded.chars().all(|c| {
        let v = c as u32;
        (0xE0061..=0xE006A).contains(&v)
            || (0xF0000..=0xFFFFF).contains(&v)
            || (0x100000..=0x10FFFF).contains(&v)
    }));
    let back = EmojiExporter::from_unicode(&encoded).expect("from_unicode");
    assert_eq!(emoji, back);
}

#[test]
fn unicode_roundtrip_embedded_in_text() {
    let emoji = full_emoji();
    let encoded = EmojiExporter::to_unicode(&emoji).expect("to_unicode");
    let wrapped = format!("hello 👋 {encoded} trailing prose");
    let back = EmojiExporter::from_unicode(&wrapped).expect("from_unicode");
    assert_eq!(emoji, back);
}

#[test]
fn empty_emoji_roundtrip() {
    // Zero blocks: header with block_count = 0, no payloads.
    let emoji = Emoji {
        name: String::new(),
        blocks: Vec::new(),
    };
    let bytes = EmojiExporter::to_binary(&emoji).expect("to_binary");
    assert_eq!(bytes.len(), 12);
    let back = EmojiExporter::from_binary(&bytes).expect("from_binary");
    assert!(back.blocks.is_empty());

    let encoded = EmojiExporter::to_unicode(&emoji).expect("to_unicode");
    assert!(encoded.is_empty());
    let back = EmojiExporter::from_unicode(&encoded).expect("from_unicode");
    assert!(back.blocks.is_empty());
}

#[test]
fn spec_quick_start_shape() {
    let emoji = EmojiBuilder::new("my-emoji")
        .description("a demo emoji")
        .add_todo("task1", "Build the thing")
        .add_image_rgba(64, 64, vec![0u8; 64 * 64 * 4])
        .with_agent_lifecycle()
        .build();
    assert_eq!(emoji.name, "my-emoji");
    assert_eq!(emoji.get_text().expect("text").description, "a demo emoji");
    assert!(emoji.get_image().is_some());
    let bytes = EmojiExporter::to_binary(&emoji).expect("to_binary");
    let back = EmojiExporter::from_binary(&bytes).expect("from_binary");
    assert_eq!(emoji, back);
}

#[test]
fn multi_frame_atlas_roundtrip_and_extract() {
    // 4x2 atlas of four 2x1 frames, each frame a distinct solid color.
    let mut rgba = vec![0u8; 4 * 2 * 4];
    for (i, px) in rgba.chunks_exact_mut(4).enumerate() {
        px[0] = (i * 60) as u8;
        px[3] = 255;
    }
    let emoji = EmojiBuilder::new("frames").add_block(Block::Img(ImageBlock {
        name: "anim".into(),
        atlas: SpriteAtlas {
            width: 4,
            height: 2,
            frame_width: 2,
            frame_height: 1,
            frame_count: 4,
            rgba,
        },
    }));
    let emoji = emoji.build();
    let bytes = EmojiExporter::to_binary(&emoji).expect("to_binary");
    let back = EmojiExporter::from_binary(&bytes).expect("from_binary");
    assert_eq!(emoji, back);

    let atlas = &back.get_image().expect("image").atlas;
    // Frame 2 sits at (x=0, y=1) → linear pixels 4 and 5 → red 4*60 = 240.
    let frame = sprites::extract_frame(atlas, 2).expect("frame 2");
    assert_eq!(frame.len(), 2 * 1 * 4);
    assert_eq!(frame[0], 240);
    assert!(sprites::extract_frame(atlas, 4).is_err());
}
