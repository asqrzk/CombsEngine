/**
 * createWorkflow: declarative multi-step pipelines over the graph engine.
 *
 * ```ts
 * const wf = createWorkflow<Doc>({
 *   steps: [
 *     { name: "draft", run: draftNode },
 *     { name: "review", run: reviewNode,
 *       loops: { max: 3, until: (s) => s.approved } },
 *   ],
 *   checks: [{ after: "draft", assert: (s) => s.draft.length > 0, message: "empty draft" }],
 * });
 * const result = await wf.invoke({ topic: "edge AI" }, { threadId: "run-1" });
 * ```
 *
 * - `loops` wraps a step in a self-loop: it re-runs until `until(state)` is
 *   true or `max` iterations are exhausted (iteration count is tracked in
 *   `state.__loop_<step>`).
 * - `checks` insert a validation gate after a step; a failed check throws
 *   (or routes to `onFail` if given).
 */

import { END, START, StateGraph } from "@combs/graph";
import type { CompiledGraph, Checkpointer, NodeFn } from "@combs/graph";

export interface WorkflowStep<S extends Record<string, unknown>> {
  name: string;
  run: NodeFn<S>;
  /** Retry policy for the step. */
  retry?: { maxAttempts: number; backoffMs?: number };
  /** Self-loop until a condition holds. */
  loops?: {
    max: number;
    until: (state: S) => boolean;
    /** Where to go when max is exhausted without success (default: continue). */
    onExhaust?: "continue" | "fail";
  };
}

export interface WorkflowCheck<S extends Record<string, unknown>> {
  after: string;
  assert: (state: S) => boolean;
  message: string;
  /** Route target on failure instead of throwing. */
  onFail?: string;
}

export interface WorkflowOptions<S extends Record<string, unknown>> {
  steps: WorkflowStep<S>[];
  checks?: WorkflowCheck<S>[];
  checkpointer?: Checkpointer;
}

export function createWorkflow<S extends Record<string, unknown>>(
  options: WorkflowOptions<S>,
  channels: { [K in keyof S]: import("@combs/graph").ChannelFactory },
): CompiledGraph<S> {
  const graph = new StateGraph<S>(channels);

  for (const step of options.steps) {
    graph.addNode(step.name, step.run, { retry: step.retry });
  }

  // Wiring: linear chain with optional self-loops.
  const names = options.steps.map((s) => s.name);
  graph.addEdge(START, names[0]);
  for (let i = 0; i < names.length; i++) {
    const step = options.steps[i];
    const next = i + 1 < names.length ? names[i + 1] : END;

    if (step.loops) {
      const counterKey = `__loop_${step.name}` as keyof S;
      // Wrap the step to count its own iterations.
      const original = step.run;
      graph.addNode(step.name, async (state, ctx) => {
        const result = await original(state, ctx);
        const count = ((state[counterKey] as number | undefined) ?? 0) + 1;
        if (result && typeof result === "object" && !("goto" in result)) {
          return { ...(result as Record<string, unknown>), [counterKey]: count } as Partial<S>;
        }
        return result;
      }, { retry: step.retry });
      graph.addConditionalEdges(step.name, (state) => {
        if (step.loops!.until(state)) return next;
        const count = (state[counterKey] as number | undefined) ?? 0;
        if (count >= step.loops!.max) {
          if (step.loops!.onExhaust === "fail") {
            throw new Error(`workflow step "${step.name}" exhausted ${step.loops!.max} loops`);
          }
          return next;
        }
        return step.name; // loop back
      });
    } else {
      // Checks after this step.
      const checks = (options.checks ?? []).filter((c) => c.after === step.name);
      if (checks.length > 0) {
        const gate = `__check_${step.name}`;
        graph.addNode(gate, (state) => {
          for (const check of checks) {
            if (!check.assert(state)) {
              if (check.onFail) return { goto: check.onFail };
              throw new Error(`workflow check failed after "${step.name}": ${check.message}`);
            }
          }
          return {};
        });
        graph.addEdge(step.name, gate);
        graph.addEdge(gate, next);
      } else {
        graph.addEdge(step.name, next);
      }
    }
  }

  return graph.compile({ checkpointer: options.checkpointer });
}
