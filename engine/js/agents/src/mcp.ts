/**
 * MCP connector: Model Context Protocol client.
 *
 * Connects to an MCP server over stdio (spawned subprocess, newline-delimited
 * JSON-RPC) or WebSocket, lists its tools and exposes them as @combs/agents
 * Tools, so remote tools drop straight into any agent's ToolRegistry.
 *
 * Implements the MCP lifecycle: initialize → notifications/initialized →
 * tools/list → tools/call.
 */

import { getLogger } from "@combs/telemetry";
import type { Tool } from "./tools.ts";
import { ToolRegistry } from "./tools.ts";

const log = getLogger("combs.mcp");

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: { code: number; message: string };
}

interface McpTransport {
  send(message: JsonRpcRequest | { jsonrpc: "2.0"; method: string }): void;
  close(): void;
  onMessage: (handler: (msg: JsonRpcResponse) => void) => void;
}

const PROTOCOL_VERSION = "2024-11-05";
const CLIENT_INFO = { name: "combs-engine", version: "0.2.0" };

/** Stdio transport: spawns a subprocess and talks JSON-RPC over pipes. */
export class StdioTransport implements McpTransport {
  private process: Deno.ChildProcess;
  private writer: WritableStreamDefaultWriter<Uint8Array>;
  private handler: (msg: JsonRpcResponse) => void = () => {};
  private buffer = "";
  private encoder = new TextEncoder();

  private constructor(command: string, args: string[], env?: Record<string, string>) {
    this.process = new Deno.Command(command, {
      args,
      env,
      stdin: "piped",
      stdout: "piped",
      stderr: "inherit",
    }).spawn();
    this.writer = this.process.stdin.getWriter();
    this.readLoop();
  }

  static spawn(command: string, args: string[] = [], env?: Record<string, string>): StdioTransport {
    return new StdioTransport(command, args, env);
  }

  private async readLoop(): Promise<void> {
    const decoder = new TextDecoder();
    const reader = this.process.stdout.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        this.buffer += decoder.decode(value, { stream: true });
        let idx: number;
        while ((idx = this.buffer.indexOf("\n")) >= 0) {
          const line = this.buffer.slice(0, idx).trim();
          this.buffer = this.buffer.slice(idx + 1);
          if (!line) continue;
          try {
            this.handler(JSON.parse(line));
          } catch {
            log.warn("mcp: dropping non-JSON line", { line: line.slice(0, 120) });
          }
        }
      }
    } catch {
      // process ended
    }
  }

  send(message: JsonRpcRequest | { jsonrpc: "2.0"; method: string }): void {
    this.writer.write(this.encoder.encode(JSON.stringify(message) + "\n"));
  }

  onMessage(handler: (msg: JsonRpcResponse) => void): void {
    this.handler = handler;
  }

  close(): void {
    try {
      this.writer.close();
    } catch { /* already closed */ }
    try {
      this.process.kill("SIGTERM");
    } catch { /* already dead */ }
  }
}

/** WebSocket transport for remote MCP servers. */
export class WebSocketTransport implements McpTransport {
  private ws: WebSocket;
  private ready: Promise<void>;

  constructor(url: string, protocols?: string[]) {
    this.ws = new WebSocket(url, protocols);
    this.ready = new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (e) => reject(e);
    });
  }

  async connect(): Promise<void> {
    await this.ready;
  }

  send(message: JsonRpcRequest | { jsonrpc: "2.0"; method: string }): void {
    this.ws.send(JSON.stringify(message));
  }

  onMessage(handler: (msg: JsonRpcResponse) => void): void {
    this.ws.onmessage = (ev) => {
      try {
        handler(JSON.parse(String(ev.data)));
      } catch {
        log.warn("mcp: dropping non-JSON ws frame");
      }
    };
  }

  close(): void {
    this.ws.close();
  }
}

/** A connected MCP client. */
export class McpClient {
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

  private constructor(private transport: McpTransport) {}

  /** Connects over stdio: `McpClient.overStdio("npx", ["-y", "@modelcontextprotocol/server-everything"])`. */
  static async overStdio(command: string, args: string[] = [], env?: Record<string, string>): Promise<McpClient> {
    const client = new McpClient(StdioTransport.spawn(command, args, env));
    await client.handshake();
    return client;
  }

  /** Connects over WebSocket. */
  static async overWebSocket(url: string): Promise<McpClient> {
    const transport = new WebSocketTransport(url);
    await transport.connect();
    const client = new McpClient(transport);
    await client.handshake();
    return client;
  }

  private async handshake(): Promise<void> {
    this.transport.onMessage((msg) => {
      const slot = this.pending.get(msg.id);
      if (!slot) return;
      this.pending.delete(msg.id);
      if (msg.error) slot.reject(new Error(`mcp error ${msg.error.code}: ${msg.error.message}`));
      else slot.resolve(msg.result);
    });
    await this.call("initialize", {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: CLIENT_INFO,
    });
    this.transport.send({ jsonrpc: "2.0", method: "notifications/initialized" });
  }

  /** JSON-RPC call with a 30s timeout. */
  call<T = unknown>(method: string, params?: Record<string, unknown>): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`mcp call ${method} timed out`));
      }, 30_000);
      this.pending.set(id, {
        resolve: (v) => {
          clearTimeout(timer);
          resolve(v as T);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
      this.transport.send({ jsonrpc: "2.0", id, method, params });
    });
  }

  /** Lists the server's tools. */
  async listTools(): Promise<{ name: string; description?: string; inputSchema?: Record<string, unknown> }[]> {
    const result = await this.call<{ tools?: { name: string; description?: string; inputSchema?: Record<string, unknown> }[] }>(
      "tools/list",
    );
    return result.tools ?? [];
  }

  /** Calls a server tool. */
  async callTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    const result = await this.call<{ content?: { type: string; text?: string }[]; isError?: boolean }>(
      "tools/call",
      { name, arguments: args },
    );
    if (result.isError) {
      const text = result.content?.map((c) => c.text ?? "").join("\n");
      throw new Error(`mcp tool ${name} failed: ${text}`);
    }
    const texts = result.content?.filter((c) => c.type === "text").map((c) => c.text) ?? [];
    return texts.length === 1 ? texts[0] : texts.join("\n");
  }

  /** Fetches server tools and registers them into a ToolRegistry. */
  async registerTools(registry: ToolRegistry, prefix = ""): Promise<Tool[]> {
    const remote = await this.listTools();
    const tools: Tool[] = remote.map((t) => ({
      name: `${prefix}${t.name}`,
      description: t.description ?? `MCP tool ${t.name}`,
      schema: t.inputSchema ?? { type: "object" },
      invoke: async (args: Record<string, unknown>) => await this.callTool(t.name, args),
    }));
    registry.registerAll(tools);
    log.info("mcp: registered tools", { count: tools.length });
    return tools;
  }

  close(): void {
    this.transport.close();
  }
}
