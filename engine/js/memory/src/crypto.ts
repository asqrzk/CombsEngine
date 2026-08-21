/**
 * At-rest crypto door for observation contents, over @combs/zerotrust's
 * keystore (per-blob data keys wrapped by a master key; decrypt
 * integrity-verifies and THROWS on tamper — the store maps that to
 * `[unreadable]`, never to wrong plaintext).
 *
 * Trade-off, by design: only observation CONTENTS encrypt. Entity
 * names, types, projects, and relations stay plaintext so indexes and
 * traversal keep working — the graph's shape is visible to a disk
 * attacker, the facts are not.
 */

// Full jsr specifier (not the workspace import-map name): this module
// is dynamically imported from OUTSIDE the engine workspace (the
// platform's COMBS_MEMORY_MOD door), where no import map is in scope.
import { decryptBlob, encryptBlob, generateMasterKey } from "jsr:@combs/zerotrust@^0.2.0";
import type { BlobCrypto, ManifestEntry } from "./types.ts";

function defaultKeyPath(): string {
  const home = Deno.env.get("HOME") ?? ".";
  return `${home}/.combs/memory/master.key`;
}

/** File-backed master key (created 0600 on first use). */
export async function fileKeyCrypto(keyPath?: string): Promise<BlobCrypto> {
  const path = keyPath ?? defaultKeyPath();
  let master: Uint8Array;
  try {
    master = await Deno.readFile(path);
    if (master.length !== 32) throw new Error(`master key at ${path} is not 32 bytes`);
  } catch (e) {
    if (!(e instanceof Deno.errors.NotFound)) throw e;
    master = generateMasterKey();
    const dir = path.substring(0, path.lastIndexOf("/"));
    if (dir) {
      await Deno.mkdir(dir, { recursive: true });
      // recursive mkdir ignores mode on existing dirs — set it plainly.
      await Deno.chmod(dir, 0o700).catch(() => {});
    }
    await Deno.writeFile(path, master, { mode: 0o600 });
  }
  return {
    async seal(data: Uint8Array): Promise<{ blob: string; entry: ManifestEntry }> {
      return await encryptBlob(master, data);
    },
    async open(blob: string, entry: ManifestEntry): Promise<Uint8Array> {
      return await decryptBlob(master, blob, entry);
    },
  };
}
