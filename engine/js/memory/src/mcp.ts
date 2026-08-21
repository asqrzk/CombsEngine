/**
 * MCP stdio SERVER: serves the knowledge graph as MCP tools to any
 * agent (same JSON-RPC 2.0 newline-delimited framing as the mesh
 * server and the @combs/agents client). Dependency-injected against
 * `GraphOps` so tests drive it without a database. Unlike the mesh
 * server, `handle` is async — the graph surface is async by contract
 * (the crypto door lives on WebCrypto).
 */

import type { GraphifyResult } from "./graphify.ts";
import type { RecallHit, RecallQuery } from "./recall.ts";
import type {
  Entity,
  EntitySpec,
  Observation,
  Relation,
  RelationSpec,
} from "./types.ts";

const PROTOCOL_VERSION = "2024-11-05";
const SERVER_INFO = { name: "combs-memory", version: "0.2.2" };

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number | null;
  result?: unknown;
  error?: { code: number; message: string };
}

/** The graph surface the server needs (structural — the bin wires it). */
export interface GraphOps {
  upsertEntities(specs: EntitySpec[]): Promise<Entity[]>;
  addObservations(entity: string, contents: string[]): Promise<Observation[]>;
  addRelations(rels: RelationSpec[]): Promise<Relation[]>;
  deleteMixed(sel: {
    entities?: string[];
    observationIds?: number[];
    relations?: RelationSpec[];
  }): Promise<number>;
  getEntity(name: string): Promise<unknown>;
  search(q: {
    query?: string;
    type?: string;
    project?: string;
    limit?: number;
  }): Promise<unknown>;
  neighbors(
    name: string,
    opts?: { depth?: number; relType?: string; direction?: string },
  ): Promise<unknown>;
  recall(q: RecallQuery): Promise<{ text: string; hits: RecallHit[] }>;
  setState(project: string, lines: string[]): Promise<void>;
  where(project?: string): Promise<string>;
  graphify(path: string, project?: string, maxFiles?: number): Promise<GraphifyResult>;
  stats(): Promise<unknown>;
}

const TOOLS = [
  {
    name: "memory_upsert_entities",
    description: "Create or update graph entities with optional initial observations",
    inputSchema: {
      type: "object",
      properties: {
        entities: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              type: { type: "string" },
              project: { type: "string" },
              caste: { type: "string", enum: ["youngling", "worker", "queen", "drone"] },
              observations: { type: "array", items: { type: "string" } },
            },
            required: ["name", "type"],
          },
        },
      },
      required: ["entities"],
    },
  },
  {
    name: "memory_add_observations",
    description: "Append fact lines to an entity",
    inputSchema: {
      type: "object",
      properties: {
        entity: { type: "string" },
        contents: { type: "array", items: { type: "string" } },
      },
      required: ["entity", "contents"],
    },
  },
  {
    name: "memory_add_relations",
    description: "Add directed typed relations between entities",
    inputSchema: {
      type: "object",
      properties: {
        relations: {
          type: "array",
          items: {
            type: "object",
            properties: {
              from: { type: "string" },
              to: { type: "string" },
              relType: { type: "string" },
            },
            required: ["from", "to", "relType"],
          },
        },
      },
      required: ["relations"],
    },
  },
  {
    name: "memory_delete",
    description: "Delete entities (cascade), observations by id, or relations",
    inputSchema: {
      type: "object",
      properties: {
        entities: { type: "array", items: { type: "string" } },
        observationIds: { type: "array", items: { type: "number" } },
        relations: {
          type: "array",
          items: {
            type: "object",
            properties: {
              from: { type: "string" },
              to: { type: "string" },
              relType: { type: "string" },
            },
            required: ["from", "to", "relType"],
          },
        },
      },
    },
  },
  {
    name: "memory_get",
    description: "Fetch one entity with its observations and 1-hop relations",
    inputSchema: {
      type: "object",
      properties: { name: { type: "string" } },
      required: ["name"],
    },
  },
  {
    name: "memory_search",
    description: "Search entities by name substring, scoped by project or type",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string" },
        project: { type: "string" },
        type: { type: "string" },
        limit: { type: "number" },
      },
    },
  },
  {
    name: "memory_neighbors",
    description: "Traverse the graph outward from an entity",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string" },
        depth: { type: "number" },
        relType: { type: "string" },
        direction: { type: "string", enum: ["out", "in", "both"] },
      },
      required: ["name"],
    },
  },
  {
    name: "memory_recall",
    description: "Ranked compact recall lines for a query (touches what it returns)",
    inputSchema: {
      type: "object",
      properties: {
        text: { type: "string" },
        project: { type: "string" },
        k: { type: "number" },
      },
      required: ["text"],
    },
  },
  {
    name: "memory_set_state",
    description: "Replace a project's current-state lines and repoint `now` at it",
    inputSchema: {
      type: "object",
      properties: {
        project: { type: "string" },
        lines: { type: "array", items: { type: "string" } },
      },
      required: ["project", "lines"],
    },
  },
  {
    name: "memory_where",
    description: "Where are we — the `now` pointer's orientation, or a named project's",
    inputSchema: {
      type: "object",
      properties: { project: { type: "string" } },
    },
  },
  {
    name: "memory_graphify",
    description: "Ingest a repository path into the graph (tracked files only)",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string" },
        project: { type: "string" },
        maxFiles: { type: "number" },
      },
      required: ["path"],
    },
  },
  {
    name: "memory_stats",
    description: "Entity/observation/relation counts, projects, and caste census",
    inputSchema: { type: "object", properties: {} },
  },
];

/** MCP server for the knowledge graph. */
export class MemoryMcpServer {
  constructor(private ops: GraphOps) {}

  /**
   * Handles one JSON-RPC message. Returns the response, or null for
   * notifications (which must not be answered).
   */
  async handle(request: JsonRpcRequest): Promise<JsonRpcResponse | null> {
    const id = request.id ?? null;
    try {
      const result = await this.dispatch(request.method, request.params ?? {});
      if (request.id === undefined) return null;
      return { jsonrpc: "2.0", id, result };
    } catch (e) {
      if (request.id === undefined) return null;
      const code = (e as { code?: number }).code ?? -32603;
      return {
        jsonrpc: "2.0",
        id,
        error: { code, message: e instanceof Error ? e.message : String(e) },
      };
    }
  }

  private async dispatch(method: string, params: Record<string, unknown>): Promise<unknown> {
    switch (method) {
      case "initialize":
        return {
          protocolVersion: PROTOCOL_VERSION,
          capabilities: { tools: {} },
          serverInfo: SERVER_INFO,
        };
      case "notifications/initialized":
        return {};
      case "tools/list":
        return { tools: TOOLS };
      case "tools/call":
        return await this.callTool(params);
      default: {
        const err = new Error(`unknown method: ${method}`);
        (err as { code?: number }).code = -32601;
        throw err;
      }
    }
  }

  private toolResult(value: unknown): unknown {
    return { content: [{ type: "text", text: JSON.stringify(value) }] };
  }

  private async callTool(params: Record<string, unknown>): Promise<unknown> {
    const name = String(params.name ?? "");
    const args = (params.arguments ?? {}) as Record<string, unknown>;
    switch (name) {
      case "memory_upsert_entities":
        return this.toolResult({
          entities: await this.ops.upsertEntities((args.entities ?? []) as EntitySpec[]),
        });
      case "memory_add_observations": {
        const entity = String(args.entity ?? "");
        if (!entity) throw new Error("memory_add_observations needs `entity`");
        return this.toolResult({
          observations: await this.ops.addObservations(
            entity,
            (args.contents ?? []) as string[],
          ),
        });
      }
      case "memory_add_relations":
        return this.toolResult({
          relations: await this.ops.addRelations((args.relations ?? []) as RelationSpec[]),
        });
      case "memory_delete":
        return this.toolResult({
          deleted: await this.ops.deleteMixed({
            entities: args.entities as string[] | undefined,
            observationIds: args.observationIds as number[] | undefined,
            relations: args.relations as RelationSpec[] | undefined,
          }),
        });
      case "memory_get": {
        const got = await this.ops.getEntity(String(args.name ?? ""));
        if (!got) throw new Error(`no entity named ${args.name}`);
        return this.toolResult(got);
      }
      case "memory_search":
        return this.toolResult({
          hits: await this.ops.search({
            query: args.query as string | undefined,
            project: args.project as string | undefined,
            type: args.type as string | undefined,
            limit: args.limit as number | undefined,
          }),
        });
      case "memory_neighbors":
        return this.toolResult(
          await this.ops.neighbors(String(args.name ?? ""), {
            depth: args.depth as number | undefined,
            relType: args.relType as string | undefined,
            direction: args.direction as string | undefined,
          }),
        );
      case "memory_recall":
        return this.toolResult(
          await this.ops.recall({
            text: String(args.text ?? ""),
            project: args.project as string | undefined,
            k: args.k as number | undefined,
          }),
        );
      case "memory_set_state": {
        const project = String(args.project ?? "");
        if (!project) throw new Error("memory_set_state needs `project`");
        await this.ops.setState(project, (args.lines ?? []) as string[]);
        return this.toolResult({ ok: true, project });
      }
      case "memory_where":
        return this.toolResult({
          text: await this.ops.where(args.project as string | undefined),
        });
      case "memory_graphify": {
        const path = String(args.path ?? "");
        if (!path) throw new Error("memory_graphify needs `path`");
        return this.toolResult(
          await this.ops.graphify(
            path,
            args.project as string | undefined,
            typeof args.maxFiles === "number" && Number.isFinite(args.maxFiles)
              ? args.maxFiles
              : undefined,
          ),
        );
      }
      case "memory_stats":
        return this.toolResult(await this.ops.stats());
      default:
        throw new Error(`unknown tool: ${name}`);
    }
  }

  /** Serves newline-delimited JSON-RPC on stdin/stdout forever. */
  async serveStdio(): Promise<void> {
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const writer = Deno.stdout.writable.getWriter();
    let buffer = "";
    for await (const chunk of Deno.stdin.readable) {
      buffer += decoder.decode(chunk, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        let response: JsonRpcResponse | null = null;
        try {
          response = await this.handle(JSON.parse(line));
        } catch {
          response = {
            jsonrpc: "2.0",
            id: null,
            error: { code: -32700, message: "parse error" },
          };
        }
        if (response) {
          await writer.write(encoder.encode(JSON.stringify(response) + "\n"));
        }
      }
    }
    writer.close();
  }
}
