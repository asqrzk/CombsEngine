/**
 * @combs/graph — agentic graph framework (LangGraph-equivalent).
 *
 * - Declarative `StateGraph` with channel/reducer state merging
 * - Superstep runner with bounded concurrency, retries, aborts
 * - Static / conditional / join edges, `Command` + `Send` control flow
 * - Checkpoints (Memory / Deno KV / SQLite): resume, time travel, HITL
 * - `ctx.interrupt()` + `interruptBefore/After` breakpoints
 * - Stream modes: values | updates | custom | interrupt | debug
 */

export {
  append,
  binaryOp,
  ephemeral,
  InvalidUpdateError,
  lastValue,
  messages,
} from "./src/channels.ts";
export type { Channel, ChannelFactory, GraphMessage, ToolCall } from "./src/channels.ts";
export {
  KvCheckpointer,
  MemoryCheckpointer,
  SqliteCheckpointer,
} from "./src/checkpoint.ts";
export type { Checkpoint, Checkpointer } from "./src/checkpoint.ts";
export {
  END,
  GraphInterrupt,
  INTERRUPT,
  isCommand,
  Send,
  START,
} from "./src/commands.ts";
export type { AgentState, Command, Interrupt } from "./src/commands.ts";
export { CompiledGraph, StateGraph } from "./src/state.ts";
export type {
  CompileOptions,
  NodeContext,
  NodeFn,
  RetryPolicy,
  RunConfig,
} from "./src/state.ts";
export { PregelRunner } from "./src/runner.ts";
export type { StateSnapshot, StreamEvent, StreamMode } from "./src/runner.ts";
