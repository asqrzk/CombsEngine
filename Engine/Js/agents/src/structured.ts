/**
 * Structured output: JSON-Schema-constrained generation.
 *
 * The current engine core does not yet do grammar-guided decoding, so this
 * implements the pragmatic loop: schema → prompt instruction → generate →
 * parse → validate → (retry with the validation error). When the Rust core
 * gains logits masking, `mode: "grammar"` will route there instead —
 * call sites don't change.
 */

export interface StructuredOutputOptions {
  /** JSON Schema the model's answer must satisfy. */
  schema: Record<string, unknown>;
  /** Max attempts including the first (default 2). */
  maxAttempts?: number;
  /** Extra instruction prepended to the schema block. */
  instruction?: string;
}

/** Builds the prompt suffix instructing JSON-only output. */
export function structuredPrompt(options: StructuredOutputOptions): string {
  return [
    options.instruction ?? "Answer with a single JSON object, and nothing else.",
    "The JSON MUST conform to this JSON Schema:",
    "```json",
    JSON.stringify(options.schema, null, 2),
    "```",
    "Do not wrap the answer in markdown fences; output raw JSON only.",
  ].join("\n");
}

/** Extracts the first JSON value from model text (fenced or bare). */
export function extractJson(text: string): unknown {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/.exec(text);
  const candidates = [fenced?.[1], text.trim()];
  const firstBrace = text.search(/[{[]/);
  if (firstBrace >= 0) candidates.push(text.slice(firstBrace));
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      return JSON.parse(candidate);
    } catch {
      // try next
    }
  }
  throw new Error("no JSON found in model output");
}

/** Minimal JSON-Schema validation (types, required, properties, enum). */
export function validateAgainstSchema(value: unknown, schema: Record<string, unknown>): string[] {
  const errors: string[] = [];
  const type = schema.type as string | undefined;
  if (type) {
    const actual = Array.isArray(value) ? "array" : value === null ? "null" : typeof value;
    const matches =
      (type === "integer" && actual === "number" && Number.isInteger(value)) ||
      actual === type ||
      (type === "number" && actual === "number");
    if (!matches) errors.push(`expected type ${type}, got ${actual}`);
  }
  if (schema.enum && Array.isArray(schema.enum)) {
    if (!schema.enum.some((e) => JSON.stringify(e) === JSON.stringify(value))) {
      errors.push(`value not in enum ${JSON.stringify(schema.enum)}`);
    }
  }
  if (type === "object" && value && typeof value === "object" && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    for (const key of (schema.required as string[]) ?? []) {
      if (!(key in obj)) errors.push(`missing required property "${key}"`);
    }
    const props = (schema.properties ?? {}) as Record<string, Record<string, unknown>>;
    for (const [key, subschema] of Object.entries(props)) {
      if (key in obj) {
        errors.push(...validateAgainstSchema(obj[key], subschema).map((e) => `${key}: ${e}`));
      }
    }
    if (schema.additionalProperties === false) {
      for (const key of Object.keys(obj)) {
        if (!(key in props)) errors.push(`unexpected property "${key}"`);
      }
    }
  }
  if (type === "array" && Array.isArray(value) && schema.items) {
    value.forEach((item, i) => {
      errors.push(
        ...validateAgainstSchema(item, schema.items as Record<string, unknown>).map(
          (e) => `[${i}]: ${e}`,
        ),
      );
    });
  }
  return errors;
}
