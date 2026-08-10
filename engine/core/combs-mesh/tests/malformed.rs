//! Malformed-input tests: every truncation and plenty of random garbage
//! must produce `Err` (or an empty decode), never a panic.

use combs_mesh::*;

/// Tiny deterministic xorshift PRNG — no dev-dependency needed.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn byte(&mut self) -> u8 {
        (self.next() & 0xFF) as u8
    }
}

fn sample_emoji() -> Emoji {
    EmojiBuilder::new("fuzz-emoji")
        .description("fuzz me")
        .add_todo("t1", "do it")
        .add_image_rgba(4, 4, vec![42u8; 4 * 4 * 4])
        .with_agent_lifecycle()
        .build()
}

#[test]
fn truncated_binaries_error_never_panic() {
    let bytes = EmojiExporter::to_binary(&sample_emoji()).expect("to_binary");
    for len in 0..bytes.len() {
        let truncated = &bytes[..len];
        // Every strict prefix must be an error (the full parse requires
        // every payload byte), and must not panic.
        assert!(
            EmojiExporter::from_binary(truncated).is_err(),
            "truncation to {len} bytes unexpectedly parsed"
        );
    }
}

#[test]
fn random_binaries_error_never_panic() {
    let mut rng = XorShift(0x1234_5678_9ABC_DEF0);
    for _ in 0..500 {
        let len = (rng.next() % 512) as usize;
        let buf: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
        // Random bytes may or may not parse; the contract is: no panic.
        let _ = EmojiExporter::from_binary(&buf);
    }
}

#[test]
fn bit_flips_error_never_panic() {
    let bytes = EmojiExporter::to_binary(&sample_emoji()).expect("to_binary");
    let mut rng = XorShift(0xDEAD_BEEF_CAFE_F00D);
    for _ in 0..500 {
        let mut mutated = bytes.clone();
        let pos = (rng.next() as usize) % mutated.len();
        mutated[pos] ^= 1 << (rng.next() % 8);
        // Bit flips may hit the name/description (still valid), so only
        // the no-panic contract is asserted here; CRC-protected regions
        // dominate and mostly error.
        let _ = EmojiExporter::from_binary(&mutated);
    }
}

#[test]
fn bad_magic_and_version() {
    let bytes = EmojiExporter::to_binary(&sample_emoji()).expect("to_binary");
    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        EmojiExporter::from_binary(&bad_magic),
        Err(MeshError::Format(_))
    ));
    let mut bad_version = bytes.clone();
    bad_version[4] = 99;
    assert!(matches!(
        EmojiExporter::from_binary(&bad_version),
        Err(MeshError::UnsupportedVersion(99))
    ));
}

#[test]
fn crc_mismatch_detected() {
    let mut bytes = EmojiExporter::to_binary(&sample_emoji()).expect("to_binary");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    assert!(matches!(
        EmojiExporter::from_binary(&bytes),
        Err(MeshError::CrcMismatch)
    ));
}

#[test]
fn truncated_unicode_envelopes_error_never_panic() {
    let encoded = EmojiExporter::to_unicode(&sample_emoji()).expect("to_unicode");
    let chars: Vec<char> = encoded.chars().collect();
    // Cut at char boundaries (a UTF-8 slice mid-char would be invalid
    // anyway — decode operates on &str).
    let original = EmojiExporter::from_unicode(&encoded).expect("decode full");
    for len in 1..chars.len() {
        let truncated: String = chars[..len].iter().collect();
        // Two legal outcomes: Err (cut mid-envelope), or Ok with a strict
        // prefix of the original blocks (cut exactly on an envelope
        // boundary). Never a panic, never invented data.
        if let Ok(decoded) = EmojiExporter::from_unicode(&truncated) {
            assert!(decoded.blocks.len() < original.blocks.len());
            assert_eq!(
                decoded.blocks,
                original.blocks[..decoded.blocks.len()].to_vec()
            );
        }
    }
}

#[test]
fn random_unicode_never_panics() {
    let mut rng = XorShift(0x0F0F_0F0F_A5A5_5A5A);
    for _ in 0..300 {
        let len = (rng.next() % 64) as usize;
        let s: String = (0..len)
            .map(|_| {
                // Mix ASCII with tag chars and plane 15/16 codepoints.
                match rng.next() % 4 {
                    0 => (b'a' + (rng.next() % 26) as u8) as char,
                    1 => char::from_u32(0xE0061 + (rng.next() % 16) as u32).unwrap_or('x'),
                    2 => char::from_u32(0xF0000 + (rng.next() % 0x10000) as u32).unwrap_or('x'),
                    _ => char::from_u32(0x100000 + (rng.next() % 0x10000) as u32).unwrap_or('x'),
                }
            })
            .collect();
        let _ = EmojiExporter::from_unicode(&s);
    }
}

#[test]
fn registry_rejects_unknown_names() {
    let dir = std::env::temp_dir().join(format!("combs-mesh-test-{}", std::process::id()));
    let registry = Registry::open_at(dir.clone()).expect("open");
    assert!(registry.resolve("does-not-exist").is_err());
    assert!(registry.resolve(&"z".repeat(64)).is_err()); // hash shape, no file
    assert!(!registry.remove("does-not-exist").expect("remove"));
    let _ = std::fs::remove_dir_all(&dir);
}
