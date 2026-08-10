/**
 * Tools: the framework-agnostic tool contract + registry.
 *
 * A tool is a plain object — no classes required. Schemas are JSON Schema,
 * which is what gets marshalled into the model's system prompt and what the
 * structured-output validator uses.
 */

export interface Tool<A = Record<string, unknown>, R = unknown> {
  name: string;
  description: string;
  /** JSON Schema for the arguments object. */
  schema: Record<string, unknown>;
  invoke(args: A, ctx: ToolContext): Promise<R> | R;
}

export interface ToolContext {
  /** Abort signal for the run. */
  signal?: AbortSignal;
  /** Free-form per-run metadata (threadId, user id, ...). */
  metadata?: Record<string, unknown>;
}

/** Helper to define a tool with inference-friendly typing. */
export function tool<A = Record<string, unknown>, R = unknown>(
  def: Tool<A, R>,
): Tool<A, R> {
  return def;
}

/** Named tool collection with prompt marshalling. */
export class ToolRegistry {
  private tools = new Map<string, Tool>();

  register(tool: Tool): this {
    if (this.tools.has(tool.name)) throw new Error(`duplicate tool: ${tool.name}`);
    this.tools.set(tool.name, tool);
    return this;
  }

  registerAll(tools: Tool[]): this {
    for (const t of tools) this.register(t);
    return this;
  }

  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  list(): Tool[] {
    return [...this.tools.values()];
  }

  get size(): number {
    return this.tools.size;
  }

  /** JSON Schema array for prompts / API calls. */
  schemas(): { name: string; description: string; parameters: Record<string, unknown> }[] {
    return this.list().map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.schema,
    }));
  }

  /** Renders the tool-use instruction block for a system prompt. */
  toPromptBlock(): string {
    if (this.tools.size === 0) return "";
    return [
      "You can call tools to accomplish the task. To call one or more tools,",
      "respond with a single JSON object (and nothing else) of the form:",
      '```json',
      '{"tool_calls": [{"name": "<tool>", "args": { ... }}]}',
      '```',
      "Available tools:",
      JSON.stringify(this.schemas(), null, 2),
    ].join("\n");
  }
}

/** A parsed tool call from model output. */
export interface ParsedToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

/**
 * Extracts tool calls from model text. Accepts a fenced ```json block or a
 * bare JSON object containing "tool_calls". Returns [] when the output is a
 * plain final answer.
 */
export function parseToolCalls(text: string): ParsedToolCall[] {
  const candidates: string[] = [];
  const fenced = /```(?:json)?\s*([\s\S]*?)```/g;
  for (const m of text.matchAll(fenced)) candidates.push(m[1]);
  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    candidates.push(text.slice(firstBrace, lastBrace + 1));
  }
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate.trim());
      const calls = parsed?.tool_calls;
      if (Array.isArray(calls) && calls.length > 0) {
        return calls.map((c: { name: string; args?: Record<string, unknown>; id?: string }, i: number) => ({
          id: c.id ?? `call_${i}_${crypto.randomUUID().slice(0, 8)}`,
          name: c.name,
          args: c.args ?? {},
        }));
      }
    } catch {
      // Not JSON — try the next candidate.
    }
  }
  return [];
}
