/** Type definitions for combs-client. */

export interface ChatMessage {
  role: string;
  content: string;
}

export interface CombsClientOptions {
  /** `combs serve` URL (required — the host app always picks the server). */
  baseUrl: string;
  /** Custom fetch — auth headers, permission gates, mocks, retries. */
  fetchImpl?: typeof fetch;
  /** Received-byte hook (per SSE chunk / per JSON body). */
  onDownload?: (bytes: number) => void;
  /** Default model id for requests. */
  model?: string;
}

export interface ChatCompletionRequest {
  messages: ChatMessage[];
  model?: string;
  maxTokens?: number;
  temperature?: number;
  signal?: AbortSignal;
}

export interface StreamCallbacks {
  onDelta?: (text: string) => void;
  onDone?: (finishReason: string) => void;
  onError?: (err: Error) => void;
}

export declare class CombsClient {
  constructor(options?: CombsClientOptions);
  readonly baseUrl: string;
  health(): Promise<boolean>;
  listModels(): Promise<string[]>;
  chatCompletion(request?: ChatCompletionRequest): Promise<string>;
  streamChatCompletion(
    request?: ChatCompletionRequest,
    callbacks?: StreamCallbacks,
  ): Promise<void>;
}

export default CombsClient;
