# combs-zerotrust

Zero-trust crypto core for [Combs Engine](https://github.com/asqrzk/CombsEngine).
Pure built-in WebCrypto — **zero dependencies**, runs in browsers, Node ≥19
and Deno. Also published to JSR as `@combs/zerotrust` (same source:
`Engine/Js/zerotrust/mod.js` — keep them byte-identical).

- **identity** — ECDSA P-256 (sign) + ECDH P-256 (encrypt) keypairs
- **seal/open** — E2E envelopes: ephemeral ECDH → HKDF-SHA256 → AES-256-GCM,
  SHA-256 of plaintext inside the ciphertext, ECDSA signature outside.
  Any tampering (signature, GCM tag, hash) throws.
- **encryptBlob/decryptBlob** — at-rest encryption: per-blob data keys
  wrapped by a master key; manifest entries `{nonce, wrappedKey, sha256}`.
- **mintToken/tokenEncrypt/tokenDecrypt** — 256-bit capability tokens for
  the sandboxed-proxy flow (token1/token2).

```js
import { generateIdentity, seal, open } from "combs-zerotrust";

const alice = await generateIdentity();
const bob = await generateIdentity();
const env = await seal({
  from: alice.fingerprint,
  fromSignPriv: alice.signPriv,
  toEncPub: bob.encPub,
  plaintext: "hello",
});
console.log(await open({ toEncPriv: bob.encPriv, fromSignPub: alice.signPub, envelope: env }));
```
