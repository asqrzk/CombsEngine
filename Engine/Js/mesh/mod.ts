/**
 * @combs/mesh — CombsMesh emoji engine client (L2).
 *
 * - `Mesh`: typed FFI client over libcombsmesh_ffi (crypto, render,
 *   registry, op_json wrappers).
 * - `unicode.ts`: pure-TS port of the plane 15/16 unicode envelope codec
 *   (no FFI — browser-safe, byte-identical to the Rust core).
 * - `EmojiMcpServer`: MCP stdio server exposing emojis as tools/resources.
 */
export {
  lastError,
  libraryCandidates,
  type MeshLib,
  openLibrary,
} from "./src/ffi.ts";
export {
  base64Decode,
  base64Encode,
  type EmojiJson,
  Mesh,
  type RegistryEntry,
} from "./src/mesh.ts";
export {
  BLOCK_TAGS,
  type BlockTag,
  decodeBlocks,
  encodeBlock,
  encodeBlocks,
  MAX_PAYLOAD,
  PLANE15_BASE,
  PLANE16_BASE,
  payloadBytes,
  SUBRANGE_SIZE,
  TAG_CHAR_BASE,
  tagChar,
  tagFromChar,
  type UnicodeBlock,
} from "./src/unicode.ts";
export { EmojiMcpServer, type MeshOps } from "./src/mcp.ts";
export { MeshPeer, sha256Hex } from "./src/peer.ts";
