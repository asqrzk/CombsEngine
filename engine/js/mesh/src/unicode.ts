/**
 * Pure-TS port of combs-mesh's `codepoint.rs` — the Unicode envelope
 * encoding (plane 15/16 PUA + tag chars). No FFI: browser and no-FFI
 * consumers can encode/decode envelopes identically to the Rust core.
 *
 * Scheme (one envelope per block, byte-for-byte identical to Rust):
 *
 *   [TAG char U+E0061+idx] [2 × plane-15 length chars] [ceil(len/2) × plane-16 data chars]
 *
 * - Tag char: `U+E0061 + block-type index` (0..10).
 * - Length: payload bytes as two 12-bit chunks, big-endian, each
 *   `U+F0000 + idx × 4096 + chunk` (plane 15 = 16 sub-ranges of 4096).
 * - Data: payload bytes, 2 bytes (big-endian u16) per codepoint at
 *   `U+100000 + u16`; odd payloads are zero-padded (the length trims it).
 *
 * Payloads are the block's serde_json bytes. NOTE for parity: serde_json
 * and JSON.stringify both emit compact JSON, but object key order must
 * match the Rust struct field order — construct payload objects in
 * declaration order when byte-parity with Rust output matters.
 */

/** Block tags in wire order (index = plane-15 sub-range). */
export const BLOCK_TAGS = [
  "txt",
  "img",
  "tdo",
  "fnc",
  "api",
  "lfc",
  "chr",
  "emo",
  "enc",
  "orc",
] as const;
export type BlockTag = (typeof BLOCK_TAGS)[number];

export const TAG_CHAR_BASE = 0xe0061;
export const PLANE15_BASE = 0xf0000;
export const PLANE16_BASE = 0x100000;
export const SUBRANGE_SIZE = 0x1000;
export const MAX_PAYLOAD = 0xff_ffff;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** A block for the codec: wire tag + payload (object → compact JSON, or pre-serialized). */
export interface UnicodeBlock {
  tag: BlockTag;
  payload: unknown;
}

function tagIndex(tag: BlockTag): number {
  const idx = BLOCK_TAGS.indexOf(tag);
  if (idx < 0) throw new Error(`unknown block tag: ${tag}`);
  return idx;
}

/** The tag char marking the start of a block of type `tag`. */
export function tagChar(tag: BlockTag): string {
  return String.fromCodePoint(TAG_CHAR_BASE + tagIndex(tag));
}

/** Inverse of tagChar; null for any other char. */
export function tagFromChar(char: string): BlockTag | null {
  const cp = char.codePointAt(0)!;
  const idx = cp - TAG_CHAR_BASE;
  return idx >= 0 && idx < BLOCK_TAGS.length ? BLOCK_TAGS[idx] : null;
}

function plane15Char(tag: BlockTag, chunk: number): string {
  if (chunk < 0 || chunk >= SUBRANGE_SIZE) {
    throw new Error(`chunk ${chunk} out of range`);
  }
  return String.fromCodePoint(PLANE15_BASE + tagIndex(tag) * SUBRANGE_SIZE + chunk);
}

function plane15Value(cp: number, tag: BlockTag): number {
  const base = PLANE15_BASE + tagIndex(tag) * SUBRANGE_SIZE;
  if (cp < base || cp >= base + SUBRANGE_SIZE) {
    throw new Error(
      `expected plane-15 length char for ${tag}, got U+${cp.toString(16).toUpperCase()}`,
    );
  }
  return cp - base;
}

function plane16Char(value: number): string {
  return String.fromCodePoint(PLANE16_BASE + value);
}

function plane16Value(cp: number): number {
  if (cp < PLANE16_BASE || cp > PLANE16_BASE + 0xffff) {
    throw new Error(`expected plane-16 data char, got U+${cp.toString(16).toUpperCase()}`);
  }
  return cp - PLANE16_BASE;
}

/** Serializes a payload the way serde_json would (compact, insertion key order). */
export function payloadBytes(payload: unknown): Uint8Array {
  return encoder.encode(typeof payload === "string" ? payload : JSON.stringify(payload));
}

/** Encodes one block: tag char + length chars + data chars. */
export function encodeBlock(tag: BlockTag, payload: unknown): string {
  const bytes = payloadBytes(payload);
  if (bytes.length > MAX_PAYLOAD) {
    throw new Error(`payload of ${bytes.length} bytes exceeds the 24-bit limit`);
  }
  let out = tagChar(tag);
  out += plane15Char(tag, (bytes.length >> 12) & 0xfff);
  out += plane15Char(tag, bytes.length & 0xfff);
  for (let i = 0; i < bytes.length; i += 2) {
    out += plane16Char((bytes[i] << 8) | (i + 1 < bytes.length ? bytes[i + 1] : 0));
  }
  return out;
}

/** Encodes blocks to the Unicode envelope string (concatenated per block). */
export function encodeBlocks(blocks: UnicodeBlock[]): string {
  return blocks.map((b) => encodeBlock(b.tag, b.payload)).join("");
}

/**
 * Decodes all block envelopes found in `s`. Non-marker chars are skipped
 * (envelopes may ride inside prose). Any *started* envelope that is
 * malformed throws — never silently corrupts.
 */
export function decodeBlocks(s: string): UnicodeBlock[] {
  const chars = Array.from(s); // code-point iteration, like Rust's chars()
  const blocks: UnicodeBlock[] = [];
  let i = 0;
  const next = (): number => {
    if (i >= chars.length) throw new Error("truncated envelope");
    return chars[i++].codePointAt(0)!;
  };
  while (i < chars.length) {
    const cp = chars[i++].codePointAt(0)!;
    const idx = cp - TAG_CHAR_BASE;
    if (idx < 0 || idx >= BLOCK_TAGS.length) continue;
    const tag = BLOCK_TAGS[idx];
    const len = (plane15Value(next(), tag) << 12) | plane15Value(next(), tag);
    const count = Math.ceil(len / 2);
    const payload = new Uint8Array(count * 2);
    for (let k = 0; k < count; k++) {
      const v = plane16Value(next());
      payload[k * 2] = v >> 8;
      payload[k * 2 + 1] = v & 0xff;
    }
    const bytes = payload.subarray(0, len);
    blocks.push({ tag, payload: JSON.parse(decoder.decode(bytes)) });
  }
  return blocks;
}
