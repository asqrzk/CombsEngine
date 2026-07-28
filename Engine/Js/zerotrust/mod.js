/**
 * @combs/zerotrust — the zero-trust crypto core for Combs Engine.
 *
 * Isomorphic: runs in browsers, Node >= 19 and Deno using ONLY built-in
 * WebCrypto (`crypto.subtle`) — zero dependencies.
 *
 * Primitives:
 *  - identity:  ECDSA P-256 (signatures) + ECDH P-256 (key agreement)
 *  - envelopes: ephemeral ECDH -> HKDF-SHA256 -> AES-256-GCM, with the
 *               SHA-256 hash of the plaintext inside the ciphertext and an
 *               ECDSA signature over the envelope outside it
 *  - keystore:  AES-256-GCM at-rest encryption, per-blob data keys wrapped
 *               by a master key, manifest entries {nonce, wrappedKey, sha256}
 *  - tokens:    256-bit capability tokens (proxy token1/token2 flow)
 *
 * All binary values crossing a boundary are base64url strings.
 */

/**
 * @typedef {object} Identity
 * @property {JsonWebKey} signPub
 * @property {JsonWebKey} signPriv
 * @property {JsonWebKey} encPub
 * @property {JsonWebKey} encPriv
 * @property {string} fingerprint
 *
 * @typedef {object} Envelope
 * @property {1} v
 * @property {string} from sender fingerprint
 * @property {JsonWebKey} eph ephemeral ECDH public key
 * @property {string} nonce base64url AES-GCM nonce
 * @property {string} ct base64url ciphertext
 * @property {string} sig base64url ECDSA signature
 *
 * @typedef {object} ManifestEntry
 * @property {string} nonce
 * @property {string} wrappedKey
 * @property {string} wrapNonce
 * @property {string} sha256
 *
 * @typedef {object} TokenRecord
 * @property {string} token
 * @property {string} agentId
 * @property {number} expires
 */

const subtle = globalThis.crypto.subtle;
const te = new TextEncoder();
const td = new TextDecoder();

// --- base64url (portable) ---------------------------------------------------

export function b64encode(bytes) {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  const b64 = typeof Buffer !== "undefined"
    ? Buffer.from(bytes).toString("base64")
    : btoa(bin);
  return b64.replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

export function b64decode(s) {
  const b64 = s.replaceAll("-", "+").replaceAll("_", "/");
  if (typeof Buffer !== "undefined") return new Uint8Array(Buffer.from(b64, "base64"));
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

export function randomBytes(n) {
  return globalThis.crypto.getRandomValues(new Uint8Array(n));
}

async function sha256(bytes) {
  return new Uint8Array(await subtle.digest("SHA-256", bytes));
}

function canon(value) {
  // deterministic JSON: sorted keys, no whitespace
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canon).join(",")}]`;
  return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canon(value[k])}`).join(",")}}`;
}

// --- identity ----------------------------------------------------------------

/**
 * Generates a root identity: ECDSA P-256 signing pair + ECDH P-256
 * agreement pair. Keys are extractable so the caller controls backup.
 */
export async function generateIdentity() {
  const sign = await subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, true, ["sign", "verify"]);
  const enc = await subtle.generateKey({ name: "ECDH", namedCurve: "P-256" }, true, ["deriveBits"]);
  const signPub = await subtle.exportKey("jwk", sign.publicKey);
  const encPub = await subtle.exportKey("jwk", enc.publicKey);
  return {
    signPub,
    signPriv: await subtle.exportKey("jwk", sign.privateKey),
    encPub,
    encPriv: await subtle.exportKey("jwk", enc.privateKey),
    fingerprint: await fingerprintOf(signPub, encPub),
  };
}

/** SHA-256 fingerprint of both public keys (display/identity binding). */
export async function fingerprintOf(signPubJwk, encPubJwk) {
  const hash = await sha256(te.encode(canon({ enc: encPubJwk, sign: signPubJwk })));
  return [...hash.slice(0, 8)].map((b) => b.toString(16).padStart(2, "0")).join(":");
}

export function importSignPriv(jwk) {
  return subtle.importKey("jwk", jwk, { name: "ECDSA", namedCurve: "P-256" }, false, ["sign"]);
}
export function importSignPub(jwk) {
  return subtle.importKey("jwk", jwk, { name: "ECDSA", namedCurve: "P-256" }, false, ["verify"]);
}
export function importEncPriv(jwk) {
  return subtle.importKey("jwk", jwk, { name: "ECDH", namedCurve: "P-256" }, false, ["deriveBits"]);
}
export function importEncPub(jwk) {
  return subtle.importKey("jwk", jwk, { name: "ECDH", namedCurve: "P-256" }, false, []);
}

// --- envelopes (agent messaging) ----------------------------------------------

const HKDF_INFO = te.encode("combs-zerotrust-v1");

async function deriveMessageKey(sharedBits, nonce) {
  const base = await subtle.importKey("raw", sharedBits, "HKDF", false, ["deriveKey"]);
  return subtle.deriveKey(
    { name: "HKDF", hash: "SHA-256", salt: nonce, info: HKDF_INFO },
    base,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}

/**
 * Seals a plaintext string for a recipient.
 * @param {object} p
 * @param {string} p.from sender id (fingerprint / agent name)
 * @param {JsonWebKey} p.fromSignPriv sender ECDSA private key
 * @param {JsonWebKey} p.toEncPub recipient ECDH public key
 * @param {string} p.plaintext
 * @returns {Promise<Envelope>} JSON-safe envelope {v, from, eph, nonce, ct, sig}
 */
export async function seal({ from, fromSignPriv, toEncPub, plaintext }) {
  const eph = await subtle.generateKey({ name: "ECDH", namedCurve: "P-256" }, true, ["deriveBits"]);
  const toPub = await importEncPub(toEncPub);
  const shared = await subtle.deriveBits({ name: "ECDH", public: toPub }, eph.privateKey, 256);
  const nonce = randomBytes(12);
  const key = await deriveMessageKey(shared, nonce);

  const plainBytes = te.encode(plaintext);
  const inner = te.encode(canon({ d: b64encode(plainBytes), h: b64encode(await sha256(plainBytes)) }));
  const ct = new Uint8Array(await subtle.encrypt({ name: "AES-GCM", iv: nonce }, key, inner));

  const ephPub = await subtle.exportKey("jwk", eph.publicKey);
  const body = { ct: b64encode(ct), eph: ephPub, from, nonce: b64encode(nonce), v: 1 };
  const signKey = await importSignPriv(fromSignPriv);
  const sig = await subtle.sign({ name: "ECDSA", hash: "SHA-256" }, signKey, te.encode(canon(body)));
  return { ...body, sig: b64encode(new Uint8Array(sig)) };
}

/**
 * Opens a sealed envelope. Throws on ANY tampering (bad signature, bad
 * GCM tag, or plaintext hash mismatch).
 * @param {object} p
 * @param {JsonWebKey} p.toEncPriv recipient ECDH private key
 * @param {JsonWebKey} p.fromSignPub sender ECDSA public key (from the
 *   peer registry — never trust a key embedded in the message)
 * @param {object} p.envelope
 * @returns {Promise<string>} plaintext
 */
export async function open({ toEncPriv, fromSignPub, envelope }) {
  const { v, from, eph, nonce, ct, sig } = envelope;
  if (v !== 1) throw new Error("zerotrust: unsupported envelope version");

  const body = { ct, eph, from, nonce, v };
  const verifyKey = await importSignPub(fromSignPub);
  const sigOk = await subtle.verify(
    { name: "ECDSA", hash: "SHA-256" },
    verifyKey,
    b64decode(sig),
    te.encode(canon(body)),
  );
  if (!sigOk) throw new Error("zerotrust: envelope signature invalid");

  const ephPub = await importEncPub(eph);
  const priv = await importEncPriv(toEncPriv);
  const shared = await subtle.deriveBits({ name: "ECDH", public: ephPub }, priv, 256);
  const key = await deriveMessageKey(shared, b64decode(nonce));

  let inner;
  try {
    inner = await subtle.decrypt({ name: "AES-GCM", iv: b64decode(nonce) }, key, b64decode(ct));
  } catch {
    throw new Error("zerotrust: envelope decryption failed (tampered)");
  }
  const { d, h } = JSON.parse(td.decode(inner));
  const plainBytes = b64decode(d);
  if (b64encode(await sha256(plainBytes)) !== h) {
    throw new Error("zerotrust: plaintext hash mismatch (tampered)");
  }
  return td.decode(plainBytes);
}

// --- keystore (at-rest encryption) ---------------------------------------------

/** Generates a random 256-bit master key (raw bytes). */
export function generateMasterKey() {
  return randomBytes(32);
}

function importAes(raw, usages) {
  return subtle.importKey("raw", raw, "AES-GCM", false, usages);
}

/**
 * Encrypts a blob for at-rest storage.
 * @param {Uint8Array | CryptoKey} masterKey raw 32 bytes, OR a CryptoKey
 *   with wrapKey usage (e.g. a non-extractable key held in IndexedDB —
 *   the raw material never touches JS).
 * @param {Uint8Array} data
 * @returns {{blob: string, entry: {nonce: string, wrappedKey: string, wrapNonce: string, sha256: string}}}
 *   blob (b64 ciphertext) to write to disk + manifest entry to store.
 */
export async function encryptBlob(masterKey, data) {
  const nonce = randomBytes(12);
  const wrapNonce = randomBytes(12);
  let dataKey;
  let wrappedKey;
  if (masterKey instanceof CryptoKey) {
    // data key is extractable transiently so it can be wrapped at rest
    dataKey = await subtle.generateKey({ name: "AES-GCM", length: 256 }, true, ["encrypt", "decrypt"]);
    wrappedKey = new Uint8Array(
      await subtle.wrapKey("raw", dataKey, masterKey, { name: "AES-GCM", iv: wrapNonce }),
    );
  } else {
    const dataKeyRaw = randomBytes(32);
    dataKey = await importAes(dataKeyRaw, ["encrypt"]);
    const master = await importAes(masterKey, ["encrypt"]);
    wrappedKey = new Uint8Array(
      await subtle.encrypt({ name: "AES-GCM", iv: wrapNonce }, master, dataKeyRaw),
    );
  }
  const blob = new Uint8Array(await subtle.encrypt({ name: "AES-GCM", iv: nonce }, dataKey, data));

  return {
    blob: b64encode(blob),
    entry: {
      nonce: b64encode(nonce),
      wrappedKey: b64encode(wrappedKey),
      wrapNonce: b64encode(wrapNonce),
      sha256: b64encode(await sha256(data)),
    },
  };
}

/**
 * Decrypts a blob and verifies its integrity hash. Throws on tampering.
 * @param {Uint8Array | CryptoKey} masterKey
 * @param {string} blob b64 ciphertext
 * @param {{nonce: string, wrappedKey: string, wrapNonce: string, sha256: string}} entry
 * @returns {Promise<Uint8Array>}
 */
export async function decryptBlob(masterKey, blob, entry) {
  let dataKey;
  try {
    if (masterKey instanceof CryptoKey) {
      dataKey = await subtle.unwrapKey(
        "raw",
        b64decode(entry.wrappedKey),
        masterKey,
        { name: "AES-GCM", iv: b64decode(entry.wrapNonce) },
        { name: "AES-GCM", length: 256 },
        false,
        ["decrypt"],
      );
    } else {
      const master = await importAes(masterKey, ["decrypt"]);
      const dataKeyRaw = new Uint8Array(
        await subtle.decrypt(
          { name: "AES-GCM", iv: b64decode(entry.wrapNonce) },
          master,
          b64decode(entry.wrappedKey),
        ),
      );
      dataKey = await importAes(dataKeyRaw, ["decrypt"]);
    }
  } catch {
    throw new Error("zerotrust: wrapped key invalid (tampered manifest)");
  }
  let data;
  try {
    data = new Uint8Array(
      await subtle.decrypt({ name: "AES-GCM", iv: b64decode(entry.nonce) }, dataKey, b64decode(blob)),
    );
  } catch {
    throw new Error("zerotrust: blob decryption failed (tampered)");
  }
  if (b64encode(await sha256(data)) !== entry.sha256) {
    throw new Error("zerotrust: blob hash mismatch (tampered)");
  }
  return data;
}

// --- capability tokens (sandboxed-proxy flow) --------------------------------------

/**
 * Mints a capability token: 32 random bytes (doubles as an AES-256 key).
 * @param {string} agentId the token is bound to
 * @param {number} ttlMs
 * @returns {{token: string, agentId: string, expires: number}}
 */
export function mintToken(agentId, ttlMs = 15 * 60 * 1000) {
  return { token: b64encode(randomBytes(32)), agentId, expires: Date.now() + ttlMs };
}

/** True when the token exists, belongs to the agent, and is unexpired. */
export function tokenValid(record, agentId) {
  return !!record && record.agentId === agentId && record.expires > Date.now();
}

/** Encrypts bytes with a token used as a raw AES-256-GCM key. */
export async function tokenEncrypt(token, data) {
  const nonce = randomBytes(12);
  const key = await importAes(b64decode(token), ["encrypt"]);
  const ct = new Uint8Array(await subtle.encrypt({ name: "AES-GCM", iv: nonce }, key, data));
  return { nonce: b64encode(nonce), ct: b64encode(ct) };
}

/** Decrypts a {nonce, ct} pair with a token key. Throws on tampering. */
export async function tokenDecrypt(token, { nonce, ct }) {
  const key = await importAes(b64decode(token), ["decrypt"]);
  try {
    return new Uint8Array(
      await subtle.decrypt({ name: "AES-GCM", iv: b64decode(nonce) }, key, b64decode(ct)),
    );
  } catch {
    throw new Error("zerotrust: token decryption failed (tampered)");
  }
}
