/**
 * Auth: first-run ECDSA keypair generation with a backup ritual.
 *
 * On first launch (when `authentication` is enabled) the user generates a
 * P-256 keypair, is shown the public fingerprint and prompted to download
 * the private key JWK backup before continuing. The keypair is stored in
 * IndexedDB. Later, saved chats and settings are tied to the key id.
 */

const DB_NAME = "combs-auth";
const STORE = "keys";
const KEY_ID = "primary";

export interface KeyIdentity {
  keyId: string;
  publicKeyJwk: JsonWebKey;
  privateKeyJwk: JsonWebKey;
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

async function idbGet(key: string): Promise<KeyIdentity | undefined> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbPut(key: string, value: KeyIdentity): Promise<void> {
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

class AuthStore {
  identity = $state<KeyIdentity | null>(null);
  ready = $state(false);

  /** Loads the existing identity (if any). */
  async init(): Promise<void> {
    try {
      this.identity = (await idbGet(KEY_ID)) ?? null;
    } catch {
      this.identity = null;
    }
    this.ready = true;
  }

  /** Generates a fresh P-256 keypair and stores it (not yet backed up). */
  async generate(): Promise<void> {
    const pair = await crypto.subtle.generateKey(
      { name: "ECDSA", namedCurve: "P-256" },
      true,
      ["sign", "verify"],
    );
    const publicKeyJwk = await crypto.subtle.exportKey("jwk", pair.publicKey);
    const privateKeyJwk = await crypto.subtle.exportKey("jwk", pair.privateKey);
    const identity: KeyIdentity = {
      keyId: crypto.randomUUID(),
      publicKeyJwk,
      privateKeyJwk,
      fingerprint: await fingerprintOf(publicKeyJwk),
      createdAt: Date.now(),
      backedUp: false,
    };
    await idbPut(KEY_ID, identity);
    this.identity = identity;
  }

  /** Downloads the private key backup and marks the backup as done. */
  async downloadBackup(): Promise<void> {
    if (!this.identity) return;
    const blob = new Blob([JSON.stringify(this.identity.privateKeyJwk, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `combs-backup-${this.identity.fingerprint.replaceAll(":", "")}.json`;
    a.click();
    URL.revokeObjectURL(url);
    this.identity = { ...this.identity, backedUp: true };
    await idbPut(KEY_ID, this.identity);
  }

  /** Wipes the identity (incognito reset). */
  async reset(): Promise<void> {
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).delete(KEY_ID);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    this.identity = null;
  }
}

export const authStore = new AuthStore();
