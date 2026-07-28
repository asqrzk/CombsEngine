/**
 * @combs/observe — WebSocket broadcast sink.
 *
 * Serializes each event to a frame and hands it to an injected `broadcast`
 * function. The runtime (the proxy's Node WS server) owns the socket set and
 * does the actual send — this sink stays runtime-free. A Control Tower tab
 * subscribes to the proxy's WS endpoint and receives every event live.
 */

import type { SinkPort } from "../ports.ts";
import type { ObsEvent } from "../types.ts";

export class WebSocketSink implements SinkPort {
  constructor(private readonly broadcast: (frame: string) => void) {}
  write(event: ObsEvent): void {
    try {
      this.broadcast(JSON.stringify(event));
    } catch {
      // never break the app
    }
  }
}
