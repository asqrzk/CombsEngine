/**
 * Pure-TS unicode codec tests (no FFI): round-trips, scheme vectors and
 * edge cases. The scheme vector test pins the exact codepoints so the TS
 * port cannot drift from combs-mesh's codepoint.rs.
 */
import { assert, assertEquals, assertThrows } from "jsr:@std/assert";
import {
  BLOCK_TAGS,
  decodeBlocks,
  encodeBlock,
  encodeBlocks,
  PLANE15_BASE,
  PLANE16_BASE,
  TAG_CHAR_BASE,
  tagChar,
  tagFromChar,
  type UnicodeBlock,
} from "../mod.ts";

Deno.test("scheme vector: exact codepoints for a fixed payload", () => {
  // payload "{}" = bytes [0x7B, 0x7D], len 2 → one data char.
  const s = encodeBlock("txt", "{}");
  const cps = Array.from(s).map((c) => c.codePointAt(0)!);
  assertEquals(cps, [
    TAG_CHAR_BASE + 0, // tag char for txt (index 0)
    PLANE15_BASE + 0 * 0x1000 + 0, // length hi (0)
    PLANE15_BASE + 0 * 0x1000 + 2, // length lo (2)
    PLANE16_BASE + 0x7b7d, // the two bytes, big-endian
  ]);
});

Deno.test("tag chars round-trip all block types", () => {
  for (const tag of BLOCK_TAGS) {
    assertEquals(tagFromChar(tagChar(tag)), tag);
  }
  assertEquals(tagFromChar("a"), null);
  assertEquals(tagFromChar("😀"), null);
});

Deno.test("round-trip: every block type", () => {
  const blocks: UnicodeBlock[] = [
    { tag: "txt", payload: { name: "n", description: "d", specs: [] } },
    {
      tag: "tdo",
      payload: {
        items: [{ key: "a", value: "task a", status: "Pending", depends_on: [] }],
      },
    },
    { tag: "emo", payload: { states: [{ name: "joy", intensity: 0.9 }] } },
    { tag: "orc", payload: { directives: [{ kind: "Note", key: "k", value: "v" }] } },
  ];
  const encoded = encodeBlocks(blocks);
  assertEquals(decodeBlocks(encoded), blocks);
});

Deno.test("edge case: empty block list encodes to empty string", () => {
  assertEquals(encodeBlocks([]), "");
  assertEquals(decodeBlocks(""), []);
});

Deno.test("edge case: odd byte counts are zero-padded and trimmed", () => {
  // '{"a":1}' is 7 bytes → 4 data chars, the last carrying a zero pad.
  const payload = { a: 1 };
  const encoded = encodeBlock("api", payload);
  const cps = Array.from(encoded);
  assertEquals(cps.length, 3 + 4);
  assertEquals(decodeBlocks(encoded), [{ tag: "api", payload }]);
});

Deno.test("edge case: empty-ish payload (minimal JSON)", () => {
  for (const payload of ["{}", "[]", '""', "0", "null"]) {
    const encoded = encodeBlock("chr", payload);
    assertEquals(decodeBlocks(encoded), [{ tag: "chr", payload: JSON.parse(payload) }]);
  }
});

Deno.test("envelopes survive embedding in prose", () => {
  const blocks: UnicodeBlock[] = [{ tag: "emo", payload: { states: [] } }];
  const wrapped = `hello 👋 ${encodeBlocks(blocks)} trailing prose`;
  assertEquals(decodeBlocks(wrapped), blocks);
});

Deno.test("malformed envelopes throw, never silently corrupt", () => {
  const good = encodeBlock("txt", "{}");
  const chars = Array.from(good);
  // Truncations after the tag char must throw.
  for (let len = 1; len < chars.length; len++) {
    assertThrows(() => decodeBlocks(chars.slice(0, len).join("")));
  }
  // Wrong sub-range length char for the tag.
  const bad = tagChar("img") + String.fromCodePoint(PLANE15_BASE + 5) + "xx";
  assertThrows(() => decodeBlocks(bad));
  // Plane-15 char where data should be.
  const badData = tagChar("txt") +
    String.fromCodePoint(PLANE15_BASE) +
    String.fromCodePoint(PLANE15_BASE + 2) +
    String.fromCodePoint(PLANE15_BASE);
  assertThrows(() => decodeBlocks(badData));
});
