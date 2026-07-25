/**
 * createRoleplayChat: multi-agent turn-taking conversations.
 *
 * ```ts
 * const chat = createRoleplayChat({
 *   agents: [
 *     { name: "sherlock", role: "detective", persona: "Brilliant, terse, observant." },
 *     { name: "watson", role: "companion", persona: "Warm, asks naive questions." },
 *   ],
 *   engine,
 *   rounds: 6,
 *   memory: await addMemory({ type: "kv", scope: "roleplay" }),
 * });
 * const { transcript } = await chat.run("Discuss who moved the artifact.");
 * ```
 *
 * Turn order is round-robin by default; pass `moderator: true` to let a
 * lightweight scheduler pick the next speaker by role relevance (keyword
 * scoring, deterministic — no extra LLM call).
 */

import { lastValue, messages, MemoryCheckpointer, StateGraph, START, END } from "@combs/graph";
import type { CompiledGraph, Checkpointer, GraphMessage } from "@combs/graph";
import type { EngineClient } from "@combs/core";
import type { MemoryStore } from "@combs/agents";
import { getLogger } from "@combs/telemetry";

const log = getLogger("combs.flows.roleplay");

export interface RoleplayAgent {
  /** Unique agent name (used in the transcript). */
  name: string;
  /** The role they play ("detective", "critic", "customer", ...). */
  role: string;
  /** Persona paragraph for the system prompt. */
  persona?: string;
  /** Per-agent engine override (default: the shared engine). */
  engine?: EngineClient;
  /** Per-agent sampling overrides. */
  sampling?: Record<string, unknown>;
}

export interface RoleplayOptions {
  agents: RoleplayAgent[];
  engine: EngineClient;
  /** Total speaker turns (default: agents.length * 3). */
  rounds?: number;
  /** Stop early when this string appears in a message. */
  stopPhrase?: string;
  /** Scenario/scene description prepended to every system prompt. */
  scene?: string;
  /** Deterministic keyword-based scheduler instead of round-robin. */
  moderator?: boolean;
  /** Long-term memory shared by all agents (recalled per turn). */
  memory?: MemoryStore;
  /** Turns of transcript each agent sees (default: all). */
  windowSize?: number;
  checkpointer?: Checkpointer;
}

export interface RoleplayState extends Record<string, unknown> {
  transcript: GraphMessage[];
  round: number;
  next: string;
  stop: boolean;
}

export interface RoleplayChat {
  graph: CompiledGraph<RoleplayState>;
  run(topic: string, config?: { threadId?: string }): Promise<RoleplayState>;
}

function scoreRelevance(agent: RoleplayAgent, topic: string): number {
  const words = new Set(topic.toLowerCase().split(/\W+/));
  let score = 0;
  for (const w of `${agent.role} ${agent.persona ?? ""}`.toLowerCase().split(/\W+/)) {
    if (words.has(w) && w.length > 3) score++;
  }
  return score;
}

export function createRoleplayChat(options: RoleplayOptions): RoleplayChat {
  const agents = options.agents;
  if (agents.length === 0) throw new Error("createRoleplayChat needs at least one agent");
  const totalRounds = options.rounds ?? agents.length * 3;

  const speakNode = async (state: RoleplayState) => {
    const agent = agents.find((a) => a.name === state.next) ?? agents[0];
    const engine = agent.engine ?? options.engine;

    const window = options.windowSize
      ? state.transcript.slice(-options.windowSize)
      : state.transcript;
    const memories = options.memory
      ? await options.memory.recall(5, { agent: agent.name })
      : [];

    const system = [
      options.scene ? `Scene: ${options.scene}` : "",
      `You are ${agent.name}, playing the role of ${agent.role}.`,
      agent.persona ?? "",
      "Stay in character. Keep replies under 120 words. Address the other participants by name.",
      memories.length > 0
        ? `What you remember:\n${memories.map((m) => `- ${m.content}`).join("\n")}`
        : "",
    ].filter(Boolean).join("\n\n");

    const llmMessages = [
      { role: "system", content: system },
      ...window.map((m) => ({
        role: m.role === "assistant" ? "user" : m.role,
        content: `${m.name ?? m.role}: ${m.content}`,
      })),
    ];

    const { text } = await engine.complete({
      messages: llmMessages,
      max_tokens: 220,
      temperature: 0.8,
      top_p: 0.95,
      ...(agent.sampling ?? {}),
    });

    if (options.memory) {
      await options.memory.remember(text.slice(0, 280), { agent: agent.name });
    }

    // Schedule the next speaker.
    let next: string;
    if (options.moderator) {
      const others = agents.filter((a) => a.name !== agent.name);
      next = others
        .map((a) => ({ a, score: scoreRelevance(a, text) }))
        .sort((x, y) => y.score - x.score)[0]?.a.name ?? agents[(state.round + 1) % agents.length].name;
    } else {
      next = agents[(state.round + 1) % agents.length].name;
    }

    const stop = options.stopPhrase ? text.includes(options.stopPhrase) : false;
    log.info("turn", { agent: agent.name, round: state.round, next });
    return {
      transcript: [{ role: "assistant", name: agent.name, content: text }],
      round: state.round + 1,
      next,
      stop,
    };
  };

  const graph = new StateGraph<RoleplayState>({
    transcript: messages(),
    round: lastValue<number>(0),
    next: lastValue<string>(agents[0].name),
    stop: lastValue<boolean>(false),
  })
    .addNode("speak", speakNode)
    .addEdge(START, "speak")
    .addConditionalEdges("speak", (state) =>
      state.stop || state.round >= totalRounds ? END : "speak"
    )
    .compile({ checkpointer: options.checkpointer ?? new MemoryCheckpointer() });

  return {
    graph,
    run: async (topic, config = {}) => {
      const seed: Partial<RoleplayState> = {
        transcript: [{ role: "user", name: "scene", content: topic }],
        round: 0,
        next: agents[0].name,
        stop: false,
      };
      return await graph.invoke(seed, config);
    },
  };
}
