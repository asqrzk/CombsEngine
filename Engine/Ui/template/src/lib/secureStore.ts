/**
 * secureStore — zero-trust localStorage. Values are AES-256-GCM encrypted
 * with a master key held as a NON-EXTRACTABLE CryptoKey in IndexedDB (the
 * raw key material never exists in JS), with the SHA-256 integrity hash
 * checked on every read. Tampered or foreign data reads as absent — never
 * as plaintext.
 */

import { decryptBlob, encryptBlob } from "@combs-edge/combs-zerotrust";
import { idbGetRaw, idbPutRaw } from "./auth.svelte";

const MASTER_ID = "storage-master";

let cached: CryptoKey | null = null;

/** Loads (or generates) the non-extractable AES master key. */
async function masterKey(): Promise<CryptoKey> {
  if (cached) return cached;
  let key = await idbGetRaw<CryptoKey>(MASTER_ID);
  if (!key) {
    key = await crypto.subtle.generateKey(
      { name: "AES-GCM", length: 256 },
      false, // non-extractable — never leaves IndexedDB as raw bytes
      ["wrapKey", "unwrapKey"],
    );
    await idbPutRaw(MASTER_ID, key);
  }
  cached = key;
  return key;
}

/** Encrypts and stores a string value. */
export async function secureSet(key: string, value: string): Promise<void> {
  const { blob, entry } = await encryptBlob(await masterKey(), new TextEncoder().encode(value));
  localStorage.setItem(key, JSON.stringify({ blob, entry }));
}

/** Reads + decrypts + verifies a value; null when absent or tampered. */
export async function secureGet(key: string): Promise<string | null> {
  const raw = localStorage.getItem(key);
  if (!raw) return null;
  try {
    const { blob, entry } = JSON.parse(raw);
    const data = await decryptBlob(await masterKey(), blob, entry);
    return new TextDecoder().decode(data);
  } catch {
    return null; // legacy/foreign key or tampered → treat as absent
  }
}

export function secureRemove(key: string): void {
  localStorage.removeItem(key);
}
