/** @combs/flows — high-level factories that hide the machinery. */

export { createWorkflow } from "./src/workflow.ts";
export type { WorkflowCheck, WorkflowOptions, WorkflowStep } from "./src/workflow.ts";
export { createRoleplayChat } from "./src/roleplay.ts";
export type {
  RoleplayAgent,
  RoleplayChat,
  RoleplayOptions,
  RoleplayState,
} from "./src/roleplay.ts";
export { addMemory, withMemory } from "./src/memory.ts";
export type { MemorySpec } from "./src/memory.ts";
