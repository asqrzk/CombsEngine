/**
 * The prebuilt ReAct agent: agent ↔ tools loop over an EngineClient,
 * expressed as a compiled StateGraph (so checkpoints, HITL, streaming and
 * time travel all apply).
 *
 * Graph shape:
 *   START → agent → (tool_calls? tools : END)
 *           tools → agent
 */

import { StateGraph, END, START, messages, MemoryCheckpointer } from "@combs/graph";
import type { CompiledGraph, GraphMessage, NodeContext } from "@combs/graph";
import type { EngineClient } from "@combs/core";
import type { Checkpointer } from "@combs/graph";
import { parseToolCalls, ToolRegistry } from "./tools.ts";
import type { Tool } from "./tools.ts";
import type { MemoryStore } from "./memory.ts";

export interface AgentState extends Record<string, unknown> {
  messages: GraphMessage[];
}

export interface ReactAgentOptions {
  /** Inference transport (FFI / remote / worker engine). */
  engine: EngineClient;
  tools?: Tool[];
  /** System prompt; the tool block is appended automatically. */
  systemPrompt?: string;
  /** Sampling params for the agent step (greedy by default for tool use). */
  sampling?: Record<string, unknown>;
  /** Loop guard: max agent↔tools rounds (default 10). */
  maxRounds?: number;
  /** Optional long-term memory: recent entries are injected into the prompt. */
  memory?: MemoryStore;
  memoryLimit?: number;
  checkpointer?: Checkpointer;
}

/** The ToolNode: executes tool calls from the last assistant message. */
export function makeToolNode(registry: ToolRegistry) {
  return async (state: AgentState): Promise<Partial<AgentState>> => {
    const last = state.messages[state.messages.length - 1];
    const calls = last?.tool_calls ?? [];
    const results: GraphMessage[] = await Promise.all(
      calls.map(async (call) => {
        const tool = registry.get(call.name);
        if (!tool) {
          return {
            role: "tool",
            tool_call_id: call.id,
            content: `error: unknown tool "${call.name}"`,
          } as GraphMessage;
        }
        try {
          const result = await tool.invoke(call.args, {});
          return {
            role: "tool",
            tool_call_id: call.id,
            content: typeof result === "string" ? result : JSON.stringify(result),
          } as GraphMessage;
        } catch (err) {
          return {
            role: "tool",
            tool_call_id: call.id,
            content: `error: ${err instanceof Error ? err.message : String(err)}`,
          } as GraphMessage;
        }
      }),
    );
    return { messages: results };
  };
}

/**
 * Creates a ready-to-run agent graph. Use `agent.invoke({ messages }, { threadId })`
 * or stream events via `agent.stream(...)`.
 */
export function createReactAgent(options: ReactAgentOptions): CompiledGraph<AgentState> {
  const registry = new ToolRegistry().registerAll(options.tools ?? []);
  const maxRounds = options.maxRounds ?? 10;
  const toolNode = makeToolNode(registry);

  const agentNode = async (state: AgentState, _ctx: NodeContext<AgentState>) => {
    const rounds = state.messages.filter((m) => m.role === "assistant").length;
    const system = [
      options.systemPrompt ?? "You are a helpful assistant.",
      registry.size > 0 ? registry.toPromptBlock() : "",
      options.memory
        ? `Long-term memories:\n${(await options.memory.recall(options.memoryLimit ?? 5))
            .map((m) => `- ${m.content}`)
            .join("\n")}`
        : "",
    ]
      .filter(Boolean)
      .join("\n\n");

    const llmMessages = [
      { role: "system", content: system },
      ...state.messages
        .filter((m) => m.role !== "system")
        .map((m) => ({
          role: m.role === "tool" ? "user" : m.role,
          content:
            m.role === "tool"
              ? `[tool result ${m.tool_call_id}]: ${m.content}`
              : m.content,
        })),
    ];

    const { text } = await options.engine.complete({
      messages: llmMessages,
      max_tokens: 512,
      temperature: 0,
      ...(options.sampling ?? {}),
    });

    if (rounds >= maxRounds) {
      return {
        messages: [{
          role: "assistant",
          content: `${text}\n\n[stopped: max rounds ${maxRounds} reached]`,
        }],
      };
    }

    const calls = parseToolCalls(text);
    if (calls.length > 0) {
      return { messages: [{ role: "assistant", content: text, tool_calls: calls }] };
    }
    return { messages: [{ role: "assistant", content: text }] };
  };

  const routeAfterAgent = (state: AgentState): string => {
    const last = state.messages[state.messages.length - 1];
    return last?.tool_calls && last.tool_calls.length > 0 ? "tools" : END;
  };

  return new StateGraph<AgentState>({ messages: messages() })
    .addNode("agent", agentNode)
    .addNode("tools", toolNode)
    .addEdge(START, "agent")
    .addConditionalEdges("agent", routeAfterAgent)
    .addEdge("tools", "agent")
    .compile({ checkpointer: options.checkpointer ?? new MemoryCheckpointer() });
}
