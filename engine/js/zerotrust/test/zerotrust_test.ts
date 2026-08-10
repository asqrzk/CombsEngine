import {
  assert,
  assertEquals,
  assertRejects,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  b64decode,
  b64encode,
  decryptBlob,
  encryptBlob,
  generateIdentity,
  generateMasterKey,
  mintToken,
  open,
  seal,
  tokenDecrypt,
  tokenEncrypt,
  tokenValid,
} from "../mod.js";

Deno.test("identity: generates both pairs + stable fingerprint", async () => {
  const id = await generateIdentity();
  assert(id.signPub.x && id.encPub.x);
  assertEquals(id.fingerprint.split(":").length, 8);
});

Deno.test("envelope: seal/open round-trip between two identities", async () => {
  const alice = await generateIdentity();
  const bob = await generateIdentity();
  const env = await seal({
    from: alice.fingerprint,
    fromSignPriv: alice.signPriv,
    toEncPub: bob.encPub,
    plaintext: "hello bob — agent data 📦",
  });
  const out = await open({ toEncPriv: bob.encPriv, fromSignPub: alice.signPub, envelope: env });
  assertEquals(out, "hello bob — agent data 📦");
});

Deno.test("envelope: tampered ciphertext is rejected", async () => {
  const alice = await generateIdentity();
  const bob = await generateIdentity();
  const env = await seal({
    from: "a",
    fromSignPriv: alice.signPriv,
    toEncPub: bob.encPub,
    plaintext: "secret",
  });
  const raw = b64decode(env.ct);
  raw[0] ^= 0xff;
  const tampered = { ...env, ct: b64encode(raw) };
  await assertRejects(
    () => open({ toEncPriv: bob.encPriv, fromSignPub: alice.signPub, envelope: tampered }),
    Error,
    "zerotrust",
  );
});

Deno.test("envelope: wrong sender key is rejected (sig check)", async () => {
  const alice = await generateIdentity();
  const mallory = await generateIdentity();
  const bob = await generateIdentity();
  const env = await seal({
    from: "a",
    fromSignPriv: alice.signPriv,
    toEncPub: bob.encPub,
    plaintext: "secret",
  });
  await assertRejects(
    () => open({ toEncPriv: bob.encPriv, fromSignPub: mallory.signPub, envelope: env }),
    Error,
    "signature",
  );
});

Deno.test("envelope: wrong recipient cannot decrypt", async () => {
  const alice = await generateIdentity();
  const bob = await generateIdentity();
  const eve = await generateIdentity();
  const env = await seal({
    from: "a",
    fromSignPriv: alice.signPriv,
    toEncPub: bob.encPub,
    plaintext: "for bob only",
  });
  await assertRejects(
    () => open({ toEncPriv: eve.encPriv, fromSignPub: alice.signPub, envelope: env }),
    Error,
  );
});

Deno.test("keystore: encrypt/decrypt round-trip + tamper detection", async () => {
  const master = generateMasterKey();
  const data = new TextEncoder().encode("model-weights-or-chat-data");
  const { blob, entry } = await encryptBlob(master, data);
  const back = await decryptBlob(master, blob, entry);
  assertEquals(new TextDecoder().decode(back), "model-weights-or-chat-data");

  const raw = b64decode(blob);
  raw[raw.length - 1] ^= 0x01;
  await assertRejects(() => decryptBlob(master, b64encode(raw), entry), Error, "zerotrust");

  await assertRejects(
    () => decryptBlob(master, blob, { ...entry, sha256: entry.sha256.replace(/.$/, "A") }),
    Error,
    "zerotrust",
  );
});

Deno.test("keystore: CryptoKey master (non-extractable path) round-trip + tamper", async () => {
  const master = await crypto.subtle.generateKey(
    { name: "AES-GCM", length: 256 },
    false, // non-extractable — raw material never touches JS
    ["wrapKey", "unwrapKey"],
  );
  const data = new TextEncoder().encode("stored with a non-extractable master");
  const { blob, entry } = await encryptBlob(master, data);
  const back = await decryptBlob(master, blob, entry);
  assertEquals(new TextDecoder().decode(back), "stored with a non-extractable master");

  const raw = b64decode(blob);
  raw[3] ^= 0xff;
  await assertRejects(() => decryptBlob(master, b64encode(raw), entry), Error, "zerotrust");
});

Deno.test("tokens: mint/validate/expiry + encrypt/decrypt + tamper", async () => {
  const rec = mintToken("agent-1", 1000);
  assert(tokenValid(rec, "agent-1"));
  assert(!tokenValid(rec, "agent-2"));
  assert(!tokenValid({ ...rec, expires: Date.now() - 1 }, "agent-1"));

  const sealed = await tokenEncrypt(rec.token, new TextEncoder().encode("request"));
  const back = await tokenDecrypt(rec.token, sealed);
  assertEquals(new TextDecoder().decode(back), "request");

  const other = mintToken("agent-1");
  await assertRejects(() => tokenDecrypt(other.token, sealed), Error, "tampered");
});
