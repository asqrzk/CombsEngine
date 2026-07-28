/**
 * Zero-trust storage middleware for the proxy.
 *
 * Every file written through /api/files is encrypted at rest with
 * AES-256-GCM (per-blob data key wrapped by the proxy master key) and
 * tracked in the manifest {name -> {nonce, wrappedKey, wrapNonce, sha256}}.
 * Every read decrypts and verifies the integrity hash — tampered data
 * never leaves the proxy.
 *
 * The master key lives only in server/master.key (mode 600), generated on
 * first run. (Passkey-PRF-wrapped master keys are a documented future
 * hardening step.)
 */

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  b64decode,
  b64encode,
  decryptBlob,
  encryptBlob,
  generateMasterKey,
} from "@combs-edge/combs-zerotrust";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const KEY_FILE = path.join(HERE, "master.key");
const MANIFEST_FILE = path.join(HERE, "manifest.json");

let masterKey = null;
let manifest = null;

/** Loads (or creates) the master key + manifest. Call once at startup. */
export async function initZeroTrust() {
  try {
    masterKey = b64decode(fs.readFileSync(KEY_FILE, "utf8").trim());
  } catch {
    masterKey = generateMasterKey();
    await fsp.writeFile(KEY_FILE, b64encode(masterKey), { mode: 0o600 });
  }
  try {
    manifest = JSON.parse(fs.readFileSync(MANIFEST_FILE, "utf8"));
  } catch {
    manifest = {};
  }
}

async function saveManifest() {
  await fsp.writeFile(MANIFEST_FILE, JSON.stringify(manifest, null, 2));
}

/** Encrypts `data` for storage as `name`; returns {blob, entry}. */
export async function sealForStorage(name, data) {
  const { blob, entry } = await encryptBlob(masterKey, data);
  manifest[name] = entry;
  await saveManifest();
  return { blob, entry };
}

/**
 * Decrypts the stored blob for `name` and verifies integrity.
 * Throws ZeroTrustError on missing manifest entry or any tampering.
 */
export async function openFromStorage(name, blobB64) {
  const entry = manifest[name];
  if (!entry) throw new Error(`zerotrust: no manifest entry for ${name} (untrusted)`);
  return decryptBlob(masterKey, blobB64, entry);
}

/** Removes a manifest entry (on file delete). */
export async function forgetStorage(name) {
  delete manifest[name];
  await saveManifest();
}

/** Manifest snapshot (for debugging/audit endpoints). */
export function manifestEntries() {
  return Object.keys(manifest);
}
