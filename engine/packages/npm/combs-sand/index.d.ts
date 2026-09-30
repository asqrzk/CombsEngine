/** Type definitions for @combs/sand — semantic abstraction of neurolinguistic determinism. */

// ── the third truth value ─────────────────────────────────────────────────
export declare const unsure: unique symbol;
export type Verdict = boolean | typeof unsure;
export declare function and(a: Verdict | Decision<any> | unknown, b: Verdict | Decision<any> | unknown): Verdict;
export declare function or(a: Verdict | Decision<any> | unknown, b: Verdict | Decision<any> | unknown): Verdict;
export declare function not(a: Verdict | Decision<any> | unknown): Verdict;
export declare function verdict(v: unknown): Verdict;
export declare function isVerdict(v: unknown): v is Verdict;

// ── declarations ──────────────────────────────────────────────────────────
export type OptionSpec = string | { means: string; not?: string[]; requires?: string[] };
export interface KindSpec { type: "kind"; name: string | null; instructions: string; options: Record<string, OptionSpec>; fallback: string | null }
export interface ScaleSpec { type: "scale"; name: string | null; instructions: string; levels: { id: string; means: string }[] }
export interface TextSpec { type: "text"; instructions: string }
export interface ShapeSpec<S = ShapeSlots> { type: "shape"; name: string | null; slots: S }
export type Kind<O extends Record<string, OptionSpec>> = { readonly [K in keyof O]: K } & { readonly spec: KindSpec };
export type Scale<L extends string> = { readonly [K in L]: K } & { readonly spec: ScaleSpec };
export type TextSlot = { readonly spec: TextSpec };
export type Shape<S extends ShapeSlots> = { readonly spec: ShapeSpec<S> };
export type SlotSpec = KindSpec | ScaleSpec | TextSpec | ShapeSpec<any>;
export type Slot = { readonly spec: SlotSpec };
export type ShapeSlots = Record<string, Slot>;
export type Extracted<S extends ShapeSlots> = {
  [K in keyof S]: S[K] extends TextSlot ? string
    : S[K] extends Kind<infer O> ? (keyof O & string) | typeof unsure
    : S[K] extends Scale<infer L> ? L | typeof unsure
    : S[K] extends Shape<infer T> ? Extracted<T>
    : never;
};

export declare function kind<const O extends Record<string, OptionSpec>>(instructions: string, options: O, opts?: { fallback?: keyof O & string; name?: string }): Kind<O>;
export declare function scale<const L extends readonly string[]>(instructions: string, levels: L, opts?: { name?: string }): Scale<L[number]>;
export declare function scale<const O extends Record<string, string>>(instructions: string, levels: O, opts?: { name?: string }): Scale<keyof O & string>;
export declare function shape<const S extends ShapeSlots>(slots: S, opts?: { name?: string }): Shape<S>;
export declare function text(instructions: string): TextSlot;
export declare function named(text: string, noun: string): Named;
export interface Named { text: string; noun: string; toString(): string }
export type State = string | Named;
export interface SettleOpts { floor?: number; gap?: number; confidence?: number; passes?: number; tokens?: number; ms?: number }
export declare function settle(opts: SettleOpts): void;

// ── decisions ─────────────────────────────────────────────────────────────
export interface CallOpts { model?: string; at?: string; signal?: AbortSignal }
export type Spec = { readonly spec: SlotSpec } | string | { means: string; not?: string[]; noun?: string | null };
export interface DecideOpts extends CallOpts { floor?: number; gap?: number; confidence?: number; noun?: string; not?: string[]; speculate?: Spec[] }
export interface AskOpts extends CallOpts { system?: string; temperature?: number; maxTokens?: number }
export interface RequestInfo { id: number; model: string; road: string | null; path: string; passes: number; tokens: number; ms: number; memo: boolean }
export declare class Decision<T extends string | boolean> {
  readonly type: "kind" | "scale" | "noul";
  readonly value: T | typeof unsure;
  readonly top: T;
  readonly p: number | null;
  readonly distribution: Record<string, number> | null;
  readonly decision: "act" | "abstain";
  readonly reason: string | null;
  readonly confidence: number | null;
  readonly margin: number | null;
  readonly expected_level?: number;
  readonly cut: boolean;
  readonly request: RequestInfo;
  readonly wire: unknown;
  is(v: T): Verdict;
  atLeast(level: T): Verdict;
  atMost(level: T): Verdict;
  above(level: T): Verdict;
  below(level: T): Verdict;
  toJSON(): unknown;
}
export type DecisionOf<X> = X extends Kind<infer O> ? Decision<keyof O & string> : X extends Scale<infer L> ? Decision<L> : Decision<boolean>;

export declare function decide<O extends Record<string, OptionSpec>>(state: State, kind: Kind<O>, opts?: DecideOpts): Promise<Decision<keyof O & string>>;
export declare function decide<L extends string>(state: State, scale: Scale<L>, opts?: DecideOpts): Promise<Decision<L>>;
export declare function decide(state: State, means: string | { means: string; not?: string[]; noun?: string | null }, opts?: DecideOpts): Promise<Decision<boolean>>;
export declare function decideAll<const S extends readonly Spec[]>(state: State, specs: S, opts?: DecideOpts): Promise<{ -readonly [I in keyof S]: DecisionOf<S[I]> }>;

export interface ClassifyRequest { state: string; questions: unknown[]; thresholds?: { confidence?: number; tau?: number } }
export declare function classify(req: ClassifyRequest, opts?: CallOpts): Promise<any>;
export declare function ask(prompt: string, opts?: AskOpts): Promise<string>;
export declare function ask<S extends ShapeSlots>(prompt: string, opts: AskOpts & { as: Shape<S> }): Promise<Extracted<S>>;
export declare function extract<S extends ShapeSlots>(state: State, shape: Shape<S>, opts?: CallOpts & { using?: "decide" | "ask"; askModel?: string }): Promise<Extracted<S>>;

// ── control ───────────────────────────────────────────────────────────────
export declare function log(...parts: unknown[]): void;
export declare function parallel<T, R>(items: readonly T[], fn: (item: T, i: number) => Promise<R> | R, opts: { limit?: number; tolerate: true; at?: string }): Promise<(R | Error)[]>;
export declare function parallel<T, R>(items: readonly T[], fn: (item: T, i: number) => Promise<R> | R, opts?: { limit?: number; tolerate?: false; at?: string }): Promise<R[]>;
export declare function race<T>(...ps: (Promise<T> | Promise<T>[])[]): Promise<T>;
export declare function timeout<T>(ms: number, p: Promise<T>): Promise<T>;
export declare function retry<T>(fn: () => Promise<T>, opts?: { times?: number; on?: (e: unknown) => boolean; at?: string }): Promise<T>;
export declare function using<T>(model: string, fn: () => Promise<T> | T): Promise<T>;

// ── the run ───────────────────────────────────────────────────────────────
export interface TraceEvent { id: string; ts: number; source: string; kind: string; name: string; traceId?: string; attrs?: Record<string, string>; input?: unknown; output?: unknown; status?: string; error?: string; severity?: string }
export interface BudgetSnapshot { limits: { passes: number | null; tokens: number | null; ms: number | null; concurrency: number }; spent: { passes: number; tokens: number; requests: number; completions: number }; ms: number; stopped: { limit: string; used: number; max: number } | null }
export interface Trace { traceId: string; file: string | null; events: TraceEvent[]; budget: BudgetSnapshot; models: { decide: string | null; ask: string | null } }
export interface Bus { publish(partial: Partial<TraceEvent> & { name: string; kind: string }): TraceEvent; events?: TraceEvent[] }
export interface RunOpts { input: string; config?: SandConfig; bus?: Bus; signal?: AbortSignal; fetchImpl?: typeof fetch; env?: (name: string) => string | undefined; layers?: SandConfig[]; log?: (line: string) => void; file?: string; traceId?: string }
export declare function run(main: (input: string) => Promise<unknown> | unknown, opts?: RunOpts): Promise<{ result: unknown; trace: Trace }>;
export declare function trace(): Trace | null;
export declare function traceLines(events: TraceEvent[]): string[];
export declare function createRing(capacity?: number): Bus & { events: TraceEvent[] };

// ── configuration ─────────────────────────────────────────────────────────
export interface ModelConfig { kind: "decision" | "chat" | string; url: string; key?: string | null | (() => Promise<string> | string); model?: string; road?: "seat" | "id-means" | "label-means" | "label-anchors" | "modernbert" | "deberta" | string; headers?: Record<string, string>; extra?: Record<string, unknown>; logprobs?: boolean | "auto"; timeoutMs?: number; concurrency?: number; name?: string }
export interface SandConfig { models?: Record<string, Partial<ModelConfig>>; default?: { decide?: string | null; ask?: string | null }; settle?: { floor?: number; gap?: number; confidence?: number }; limits?: { passes?: number | null; tokens?: number | null; ms?: number | null; concurrency?: number }; net?: string[] }
export interface ResolvedConfig { models: Record<string, ModelConfig & { name: string; unavailable?: string }>; default: { decide: string | null; ask: string | null }; settle: SettleOpts; limits: { passes: number | null; tokens: number | null; ms: number | null; concurrency: number }; net: string[] }
export declare function configure(cfg: SandConfig): void;
export declare function setPorts(p: { env?: (name: string) => string | undefined; readFile?: (path: string) => Promise<string | null>; fetchImpl?: typeof fetch; layers?: SandConfig[]; log?: (line: string) => void }): void;
export declare function resolveConfig(p: { env: (name: string) => string | undefined; layers?: SandConfig[]; explicit?: SandConfig | null; local?: ModelConfig | null }): ResolvedConfig;
export declare function loadLayers(p: { env: (name: string) => string | undefined; readFile: (path: string) => Promise<string | null>; cwd?: string; home?: string | null }): Promise<SandConfig[]>;
export declare function mergeConfig(...layers: (SandConfig | null | undefined)[]): SandConfig;
export declare function deriveNet(models: Record<string, { url?: string }>, written?: string[] | null): string[];
export declare function hostOf(url: string): string | null;
export declare function questionOf(spec: SlotSpec | { type: "noul"; means: string; not: string[]; noun?: string | null }, settle?: SettleOpts): Record<string, unknown>;
export declare function normalizeSpec(spec: Spec, opts?: { not?: string[]; noun?: string | null; at?: string }): SlotSpec | { type: "noul"; means: string; not: string[]; noun: string | null };
export declare function idsOf(spec: SlotSpec | { type: string }): string[];
export declare function enumSchema(spec: SlotSpec): Record<string, unknown>;
export declare function shapeSchema(shape: ShapeSpec<any>, opts?: { only?: string[] | null }): Record<string, unknown>;
export declare function checkSchema(schema: unknown, path?: string): void;
export declare function promptOf(question: Record<string, unknown>): string;
export declare function extractionPrompt(shape: ShapeSpec<any>, only?: string[] | null): string;
export declare function questionKey(question: Record<string, unknown>): string;
export declare function canonical(value: unknown): string;
export declare function slug(s: string): string;
export declare const kindRegistry: Map<string, (model: ModelConfig & { name: string }, ports: { fetchImpl?: typeof fetch; bus?: Bus }) => ModelAdapter>;
export declare class Budget { constructor(limits?: Partial<BudgetSnapshot["limits"]>, emit?: ((name: string, kind: string, attrs: Record<string, unknown>) => void) | null); limits: BudgetSnapshot["limits"]; spent: BudgetSnapshot["spent"]; narrow(opts: { passes?: number; tokens?: number; ms?: number }): void; check(estimate?: { passes?: number; tokens?: number }, at?: string | null): void; record(spent?: { passes?: number; tokens?: number; completions?: number }): void; snapshot(): BudgetSnapshot }
export declare class Semaphore { constructor(permits: number); acquire(): Promise<void>; release(): void; run<T>(fn: () => Promise<T>): Promise<T> }
export declare function clampAttrs(attrs?: Record<string, unknown>): Record<string, string>;
export declare function refusal(modelName: string, status: number, json: unknown, text: string, ctx?: { at?: string }): SandError;
export declare function answerFrom(q: Record<string, any>, ids: string[], r: { content: string; logprobs: unknown[] | null }, settle: SettleOpts): Record<string, unknown>;
export declare class Batcher { constructor(ctx: unknown) }
export declare function tokenize(src: string): { kind: string; text: string; line: number; col: number; start: number; end: number; path?: string }[];
export declare function pairBrackets(toks: unknown[]): { open: Map<number, number>; close: Map<number, number> };
export declare function isTrivia(t: { kind: string }): boolean;
export declare function zeroConfig(env: (name: string) => string | undefined): SandConfig;
export declare const DEFAULT_LIMITS: { passes: number; tokens: number; ms: number; concurrency: number };
export declare const ROADS: Record<string, string>;

export interface ModelAdapter {
  kind: string;
  name?: string;
  road?: string | null;
  path?: string;
  caps: { batch: number; perRequest: boolean; noul: boolean; abstain: "wire" | "client"; schema: boolean; logprobs: boolean | "auto"; text: boolean };
  decide(req: { state: string; questions: unknown[]; settle?: SettleOpts; thresholds?: Record<string, number> }, ctx: { at?: string | null; signal?: AbortSignal | null; traceId?: string }): Promise<{ answers: any[]; usage?: unknown; passes?: number; tokens?: number; completions?: number; raw?: unknown; ms?: number; passesCharged?: number | null; balance?: number | null; requestId?: string | null }>;
  ask?(req: { messages: { role: string; content: string }[]; schema?: unknown; maxTokens?: number; temperature?: number }, ctx: { at?: string | null; signal?: AbortSignal | null }): Promise<{ text: string; json?: unknown; tokens?: number; completions?: number; raw?: unknown; ms?: number }>;
  extract?(state: string, shape: ShapeSpec, only: string[] | null, ctx: { at?: string | null; signal?: AbortSignal | null }): Promise<{ record: Record<string, unknown>; tokens?: number; completions?: number; raw?: unknown; ms?: number }>;
}
export declare function registerKind(name: string, factory: (model: ModelConfig & { name: string }, ports: { fetchImpl?: typeof fetch; bus?: Bus }) => Partial<ModelAdapter> & Pick<ModelAdapter, "kind" | "caps" | "decide">): void;
export declare function decisionAdapter(model: ModelConfig & { name: string }, ports?: { fetchImpl?: typeof fetch }): ModelAdapter;
export declare function chatAdapter(model: ModelConfig & { name: string }, ports?: { fetchImpl?: typeof fetch }): ModelAdapter;

// ── errors ────────────────────────────────────────────────────────────────
export declare class SandError extends Error {
  readonly code: string;
  readonly text: string;
  readonly at: string | null;
  readonly line: number | null;
  readonly col: number | null;
  readonly status: number | null;
  readonly api: unknown;
  constructor(code: string, text: string, where?: { line?: number; col?: number; at?: string; status?: number; api?: unknown; cause?: unknown });
}

// ── the compiler ──────────────────────────────────────────────────────────
export interface Warning { code: string; at: string; message: string }
export interface Plan {
  sand: string; file: string; hash: string; lines: number;
  declarations: { kinds: { name: string; at: string; options: string[]; fallback: string | null }[]; scales: { name: string; at: string; levels: string[] }[]; shapes: { name: string; at: string; slots: Record<string, string> }[] };
  settle: { at: string; passes: boolean } | null;
  sites: { id: string; at: string; text: string; how: string; binding: string; model: string; loops: string[]; batch: number | null }[];
  batches: { id: number; at: string; binding: string; sites: string[]; speculative: string[]; speculated?: boolean; questions: number; loops: string[]; conditional: boolean; model: string }[];
  calls: { at: string; verb: string; binding?: string; spec?: string | null; using?: string | null; loops: string[]; conditional: boolean; speculative?: string[] }[];
  loops: { at: string; kind: string }[];
  cost: { min: { passes: number; completions: number }; max: { passes: number; completions: number }; loops: string[]; unbounded: boolean; text: string };
  warnings: Warning[];
}
export interface TransformOptions { importStyle?: "jsr" | "relative" | "global"; importPath?: string; file?: string }
export declare function transform(src: string, opts?: TransformOptions): { code: string; plan: Plan; warnings: Warning[] };
export declare function plan(src: string, opts?: TransformOptions): Plan;
export declare const ASYNC_VERBS: string[];
export declare const SYNC_VERBS: string[];
export declare const GENERIC_CONTRARIES: string[];
export declare const PROLOGUE_LINES: number;
