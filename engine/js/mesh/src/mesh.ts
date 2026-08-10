/**
 * The Mesh client: typed wrappers over libcombsmesh_ffi — direct C symbols
 * for init/crypto, the `combsmesh_op_json` escape hatch for everything
 * else (build/encode/registry/render).
 */

import {
  cstring,
  lastError,
  type MeshLib,
  openLibrary,
  outSlots,
  readOwnedBytes,
  readOwnedCString,
} from "./ffi.ts";
import type { UnicodeBlock } from "./unicode.ts";

/** An emoji as JSON (`{name, blocks}` — the op_json wire shape). */
export interface EmojiJson {
  name: string;
  blocks: unknown[];
}

/** One registry entry (`registry_list` / `registry_resolve`). */
export interface RegistryEntry {
  name: string;
  hash: string;
  path: string;
  bytes: number;
}

/** btoa/atob-based base64 (zero-dep; Deno has both globals). */
export function base64Encode(bytes: Uint8Array): string {
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

/** Decodes base64 to bytes. */
export function base64Decode(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/** A CombsMesh engine handle (native library + process-wide keyring). */
export class Mesh {
  private constructor(private lib: MeshLib) {}

  /** Opens the library (see openLibrary for the resolution order). */
  static open(libraryPath?: string): Mesh {
    return new Mesh(openLibrary(libraryPath));
  }

  /** Initializes the keyring; no key = random master. */
  init(key?: Uint8Array): void {
    const rc = this.lib.symbols.combsmesh_init(
      key ? Deno.UnsafePointer.of(key) : null,
      BigInt(key?.length ?? 0),
    );
    if (rc !== 0) throw new Error(`combsmesh_init failed: ${lastError(this.lib)}`);
  }

  /** Encrypts a blob (AES-256-GCM, nonce-prefixed). Requires init(). */
  encrypt(data: Uint8Array): Uint8Array {
    return this.crypt(data, true);
  }

  /** Decrypts a blob produced by encrypt(). */
  decrypt(data: Uint8Array): Uint8Array {
    return this.crypt(data, false);
  }

  private crypt(data: Uint8Array, encrypt: boolean): Uint8Array {
    const { outPtr, outLen } = outSlots();
    const fn = encrypt
      ? this.lib.symbols.combsmesh_encrypt_memory
      : this.lib.symbols.combsmesh_decrypt_memory;
    const rc = fn(
      Deno.UnsafePointer.of(data),
      BigInt(data.length),
      Deno.UnsafePointer.of(outPtr),
      Deno.UnsafePointer.of(outLen),
    );
    if (rc !== 0) {
      throw new Error(
        `${encrypt ? "encrypt" : "decrypt"}_memory failed: ${lastError(this.lib)}`,
      );
    }
    return readOwnedBytes(this.lib, outPtr, outLen);
  }

  /**
   * Renders frame `frame` of the emoji in `cmse` (a .cmse binary) to RGBA8
   * pixels. Goes through the `render` op so dimensions ride along.
   */
  renderSprite(
    cmse: Uint8Array,
    frame = 0,
  ): { rgba: Uint8Array; width: number; height: number } {
    const res = this.opJson({
      op: "render",
      binary_b64: base64Encode(cmse),
      frame,
    }) as { rgba_b64: string; width: number; height: number };
    return { rgba: base64Decode(res.rgba_b64), width: res.width, height: res.height };
  }

  /** Shuts the library down, zeroizing the master key. */
  close(): void {
    this.lib.symbols.combsmesh_shutdown();
    this.lib.close();
  }

  // ---- typed op wrappers (combsmesh_op_json) ----

  /** Raw op call: `{"op": ...}` → parsed JSON response. */
  opJson(request: Record<string, unknown>): unknown {
    const ptr = this.lib.symbols.combsmesh_op_json(
      Deno.UnsafePointer.of(cstring(JSON.stringify(request))),
    );
    if (!ptr) throw new Error(`op ${request.op} failed: ${lastError(this.lib)}`);
    return JSON.parse(readOwnedCString(this.lib, ptr));
  }

  /** Builds an emoji; returns emoji JSON + binary + unicode envelope. */
  buildEmoji(input: {
    name: string;
    description?: string;
    blocks?: UnicodeBlock[] | unknown[];
  }): { emoji: EmojiJson; binary: Uint8Array; unicode: string } {
    const res = this.opJson({ op: "build", ...input }) as {
      emoji: EmojiJson;
      binary_b64: string;
      unicode: string;
    };
    return {
      emoji: res.emoji,
      binary: base64Decode(res.binary_b64),
      unicode: res.unicode,
    };
  }

  /** Parses a .cmse binary into emoji JSON. */
  emojiFromBinary(binary: Uint8Array): EmojiJson {
    const res = this.opJson({ op: "from_binary", binary_b64: base64Encode(binary) }) as {
      emoji: EmojiJson;
    };
    return res.emoji;
  }

  /** Encodes emoji JSON to the unicode envelope string. */
  toUnicode(emoji: EmojiJson): string {
    const res = this.opJson({ op: "to_unicode", emoji }) as { unicode: string };
    return res.unicode;
  }

  /** Decodes every block envelope found in `unicode`. */
  fromUnicode(unicode: string): EmojiJson {
    const res = this.opJson({ op: "from_unicode", unicode }) as { emoji: EmojiJson };
    return res.emoji;
  }

  /** Registers a .cmse binary in the content-addressed registry. */
  registryRegister(binary: Uint8Array, name?: string): string {
    const res = this.opJson({
      op: "registry_register",
      binary_b64: base64Encode(binary),
      ...(name ? { name } : {}),
    }) as { hash: string };
    return res.hash;
  }

  /** Resolves a name or sha256 hash to emoji JSON + binary. */
  registryResolve(nameOrHash: string): { emoji: EmojiJson; binary: Uint8Array } {
    const res = this.opJson({ op: "registry_resolve", name_or_hash: nameOrHash }) as {
      emoji: EmojiJson;
      binary_b64: string;
    };
    return { emoji: res.emoji, binary: base64Decode(res.binary_b64) };
  }

  /** Lists registry entries. */
  registryList(): RegistryEntry[] {
    const res = this.opJson({ op: "registry_list" }) as { entries: RegistryEntry[] };
    return res.entries;
  }
}
