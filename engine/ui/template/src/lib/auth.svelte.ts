/**
 * Auth: first-run keypair generation with a backup ritual + passkey state.
 *
 * Storage model (best practice for browser key custody):
 * - Identity keys (ECDSA P-256 signing + ECDH P-256 encryption) are
 *   generated EXTRACTABLE, but the raw private JWK lives in IndexedDB only
 *   until the user completes the backup download.
 * - After backup, private keys are re-imported as NON-EXTRACTABLE
 *   CryptoKey objects (raw material never touches JS again) and the JWKs
 *   are wiped from storage. IndexedDB structured-clones CryptoKey handles
 *   safely — no plaintext keys at rest anywhere.
 * - Cookies are deliberately NOT used: they are sent to a server on every
 *   request and are meant for session tokens, not key custody.
 */

const DB_NAME = "combs-auth";
const STORE = "keys";
const KEY_ID = "primary";
const HANDLES_ID = "primary-handles";

export interface KeyIdentity {
  keyId: string;
  publicKeyJwk: JsonWebKey;
  encPubJwk?: JsonWebKey;
  /** Present ONLY between generation and backup completion. */
  privateKeyJwk?: JsonWebKey;
  encPrivJwk?: JsonWebKey;
  fingerprint: string;
  createdAt: number;
  backedUp: boolean;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** Generic raw get (CryptoKey-safe) — shared with secureStore. */
export async function idbGetRaw<T>(key: string): Promise<T | undefined> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).get(key);
    req.onsuccess = () => resolve(req.result as T | undefined);
    req.onerror = () => reject(req.error);
  });
}

/** Generic raw put (CryptoKey-safe) — shared with secureStore. */
export async function idbPutRaw(key: string, value: unknown): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function fingerprintOf(publicKeyJwk: JsonWebKey): Promise<string> {
  const data = new TextEncoder().encode(JSON.stringify(publicKeyJwk));
  const hash = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(hash)]
    .slice(0, 8)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join(":");
}

/** Re-imports private JWKs as non-extractable CryptoKeys and wipes the JWKs. */
async function sealIdentity(identity: KeyIdentity): Promise<KeyIdentity> {
  if (!identity.privateKeyJwk || !identity.encPrivJwk) return identity;
  const signKey = await crypto.subtle.importKey(
    "jwk",
    identity.privateKeyJwk,
    { name: "ECDSA", namedCurve: "P-256" },
    false, // non-extractable
    ["sign"],
  );
  const encKey = await crypto.subtle.importKey(
    "jwk",
    identity.encPrivJwk,
    { name: "ECDH", namedCurve: "P-256" },
    false, // non-extractable
    ["deriveBits"],
  );
  await idbPutRaw(HANDLES_ID, { signKey, encKey });
  const sealed: KeyIdentity = { ...identity };
  delete sealed.privateKeyJwk;
  delete sealed.encPrivJwk;
  return sealed;
}

class AuthStore {
  identity = $state<KeyIdentity | null>(null);
  ready = $state(false);
  /** Whether a device passkey is registered with the proxy. */
  passkeyRegistered = $state(false);
  /** User chose to continue without a passkey (WebAuthn unsupported). */
  passkeySkipped = $state(false);

  /** Loads the existing identity (if any), migrating legacy plaintext keys. */
  async init(): Promise<void> {
    try {
      let identity = (await idbGetRaw<KeyIdentity>(KEY_ID)) ?? null;
      if (identity && !identity.encPubJwk) {
        // legacy pre-ECDH record: add an encryption pair
        const enc = await crypto.subtle.generateKey(
          { name: "ECDH", namedCurve: "P-256" },
          true,
          ["deriveBits"],
        );
        identity = {
          ...identity,
          encPubJwk: await crypto.subtle.exportKey("jwk", enc.publicKey),
          encPrivJwk: await crypto.subtle.exportKey("jwk", enc.privateKey),
        };
      }
      if (identity?.backedUp && identity.privateKeyJwk) {
        // legacy plaintext record: seal it now
        identity = await sealIdentity(identity);
        await idbPutRaw(KEY_ID, identity);
      }
      this.identity = identity;
    } catch {
      this.identity = null;
    }
    try {
      // Passkey state is cached per tab-session: different ports/tabs have
      // separate sessionStorage, and once a device passkey exists we never
      // ask for creation again (the credential store is global on the
      // proxy, so a status hit confirms it across apps anyway).
      const { passkeyStatus } = await import("./passkey");
      const flagged = sessionStorage.getItem("combs.passkey") === "1";
      this.passkeyRegistered = flagged || (await passkeyStatus());
      if (this.passkeyRegistered) sessionStorage.setItem("combs.passkey", "1");
    } catch {
      this.passkeyRegistered = false;
    }
    this.ready = true;
  }

  /** Generates a fresh keypair (extractable until the backup is done). */
  async generate(): Promise<void> {
    const pair = await crypto.subtle.generateKey(
      { name: "ECDSA", namedCurve: "P-256" },
      true,
      ["sign", "verify"],
    );
    const enc = await crypto.subtle.generateKey(
      { name: "ECDH", namedCurve: "P-256" },
      true,
      ["deriveBits"],
    );
    const publicKeyJwk = await crypto.subtle.exportKey("jwk", pair.publicKey);
    const identity: KeyIdentity = {
      keyId: crypto.randomUUID(),
      publicKeyJwk,
      privateKeyJwk: await crypto.subtle.exportKey("jwk", pair.privateKey),
      encPubJwk: await crypto.subtle.exportKey("jwk", enc.publicKey),
      encPrivJwk: await crypto.subtle.exportKey("jwk", enc.privateKey),
      fingerprint: await fingerprintOf(publicKeyJwk),
      createdAt: Date.now(),
      backedUp: false,
    };
    await idbPutRaw(KEY_ID, $state.snapshot(identity));
    this.identity = identity;
  }

  /** Downloads the private key backup, then seals the stored identity. */
  async downloadBackup(): Promise<void> {
    if (!this.identity) return;
    const plain = $state.snapshot(this.identity);
    const blob = new Blob([JSON.stringify(plain.privateKeyJwk, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `combs-backup-${plain.fingerprint.replaceAll(":", "")}.json`;
    a.click();
    URL.revokeObjectURL(url);

    // Backup done: private keys become non-extractable CryptoKey handles,
    // plaintext JWKs are wiped from storage.
    const sealed = await sealIdentity({ ...plain, backedUp: true });
    await idbPutRaw(KEY_ID, sealed);
    this.identity = sealed;
  }

  /** Wipes the identity (full reset). */
  async reset(): Promise<void> {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).delete(KEY_ID);
      tx.objectStore(STORE).delete(HANDLES_ID);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    this.identity = null;
  }
}

export const authStore = new AuthStore();
