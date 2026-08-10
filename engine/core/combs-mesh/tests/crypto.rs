//! Crypto tests: AEAD round-trips, HKDF determinism, tamper/wrong-key
//! detection, and encrypted binary export/import.

use combs_mesh::*;

const KEY: &[u8] = b"test master key, 32 bytes long!!!";

#[test]
fn encrypt_decrypt_roundtrip_both_algorithms() {
    let keyring = KeyRing::new(Some(KEY));
    let data = b"attack at dawn";
    for algo in [
        EncryptionAlgorithm::Aes256Gcm,
        EncryptionAlgorithm::ChaCha20Poly1305,
    ] {
        let ct = keyring.encrypt(data, algo).expect("encrypt");
        assert_ne!(ct, data);
        assert!(ct.len() > data.len() + 12); // nonce + tag overhead
        let pt = keyring.decrypt(&ct, algo).expect("decrypt");
        assert_eq!(pt, data);
    }
}

#[test]
fn hkdf_determinism() {
    let a = KeyRing::new(Some(KEY));
    let b = KeyRing::new(Some(KEY));
    let k1 = a.subkey(DEFAULT_HKDF_INFO).expect("subkey");
    let k2 = b.subkey(DEFAULT_HKDF_INFO).expect("subkey");
    assert_eq!(&k1[..], &k2[..], "same master + info → same subkey");
    let k3 = a.subkey("other-purpose").expect("subkey");
    assert_ne!(&k1[..], &k3[..], "different info → different subkey");
}

#[test]
fn random_keyring_when_no_master() {
    let a = KeyRing::new(None);
    let b = KeyRing::new(None);
    let ka = a.subkey(DEFAULT_HKDF_INFO).expect("subkey");
    let kb = b.subkey(DEFAULT_HKDF_INFO).expect("subkey");
    assert_ne!(&ka[..], &kb[..], "random masters differ");
}

#[test]
fn tampered_ciphertext_fails() {
    let keyring = KeyRing::new(Some(KEY));
    let mut ct = keyring
        .encrypt(b"payload", EncryptionAlgorithm::Aes256Gcm)
        .expect("encrypt");
    let last = ct.len() - 1;
    ct[last] ^= 0x01;
    assert!(keyring.decrypt(&ct, EncryptionAlgorithm::Aes256Gcm).is_err());
    // Also: truncated below the nonce.
    assert!(keyring.decrypt(&ct[..8], EncryptionAlgorithm::Aes256Gcm).is_err());
}

#[test]
fn wrong_key_fails() {
    let ct = KeyRing::new(Some(KEY))
        .encrypt(b"payload", EncryptionAlgorithm::ChaCha20Poly1305)
        .expect("encrypt");
    let other = KeyRing::new(Some(b"a different master key........"));
    assert!(other.decrypt(&ct, EncryptionAlgorithm::ChaCha20Poly1305).is_err());
}

#[test]
fn encrypted_binary_export_import() {
    let emoji = EmojiBuilder::new("secret-emoji")
        .description("plaintext description")
        .add_image_rgba(4, 4, vec![7u8; 4 * 4 * 4])
        .encryption(EncryptionBlock {
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            apply_to: vec![BlockTag::Img],
        })
        .build();
    let keyring = KeyRing::new(Some(KEY));

    let bytes = EmojiExporter::to_binary_encrypted(&emoji, &keyring).expect("export");
    // Encrypted at rest: the raw atlas bytes must not appear verbatim.
    assert!(!bytes.windows(16).any(|w| w == [7u8; 16]));
    // Header flag bit0 set.
    assert_eq!(bytes[6] & 1, 1);

    // Without a keyring the container refuses to decode.
    assert!(EmojiExporter::from_binary(&bytes).is_err());
    // With the wrong key it fails too.
    let wrong = KeyRing::new(Some(b"wrong master key................"));
    assert!(EmojiExporter::from_binary_decrypted(&bytes, &wrong).is_err());
    // With the right key it round-trips.
    let back = EmojiExporter::from_binary_decrypted(&bytes, &keyring).expect("import");
    assert_eq!(emoji, back);
}

#[test]
fn global_keyring_lifecycle() {
    crypto::shutdown(); // clean slate (tests share the process)
    assert!(crypto::global().is_err());
    crypto::init(Some(KEY)).expect("init");
    let kr = crypto::global().expect("global");
    let ct = kr.encrypt(b"x", EncryptionAlgorithm::Aes256Gcm).expect("enc");
    assert_eq!(
        kr.decrypt(&ct, EncryptionAlgorithm::Aes256Gcm).expect("dec"),
        b"x"
    );
    crypto::shutdown();
    assert!(crypto::global().is_err());
}

#[test]
fn default_engine_contract() {
    let engine = DefaultEngine::new();
    // Crypto before init → NotInitialized.
    assert!(matches!(
        engine.encrypt_memory(b"x"),
        Err(EngineError::NotInitialized)
    ));
    engine.init(KEY).expect("init");
    let ct = engine.encrypt_memory(b"blob").expect("encrypt");
    assert_eq!(engine.decrypt_memory(&ct).expect("decrypt"), b"blob");
    // infer is unsupported without the `engine` feature.
    assert!(matches!(
        engine.infer("hi"),
        Err(EngineError::Unsupported(_))
    ));
    // render_sprite works standalone.
    let emoji = EmojiBuilder::new("e")
        .add_image_rgba(2, 2, vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 9, 9, 9, 255])
        .build();
    let frame = engine.render_sprite(&emoji, 0).expect("render");
    assert_eq!(frame.len(), 2 * 2 * 4);
    engine.shutdown().expect("shutdown");
    assert!(matches!(
        engine.decrypt_memory(&ct),
        Err(EngineError::NotInitialized)
    ));
}
