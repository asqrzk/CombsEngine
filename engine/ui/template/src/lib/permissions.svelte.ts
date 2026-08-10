/**
 * Permissions — the frontend half. All grant state and ALL enforcement
 * live in the backend proxy (server/proxy.mjs); this store only renders
 * the ask-dialog and forwards the user's decision to the proxy, which
 * persists it and applies it to every relayed request / file write.
 */

export type Grant = "once" | "session" | "always" | "deny";

export type PermissionScope =
  | "network:download" // downloading models over the internet
  | "network:inference" // talking to the inference server
  | "network:internet" // agent web access via sandboxed proxy (token1/token2)
  | "system:subprocess" // starting local engine/agent subprocesses
  | "storage:cache" // caching data on this device
  | "storage:chats"; // persisting chat sessions

interface PendingRequest {
  scope: PermissionScope;
  detail: string;
  resolve: (grant: Grant) => void;
}

class PermissionStore {
  /** Pending permission request shown in the dialog (reactive). */
  pending = $state<PendingRequest | null>(null);

  /**
   * Shows the dialog, POSTs the decision to the backend proxy, and
   * resolves true when this attempt may proceed (caller then retries).
   */
  async ask(scope: PermissionScope, detail: string): Promise<boolean> {
    const grant = await new Promise<Grant>((resolve) => {
      this.pending = { scope, detail, resolve };
    });
    this.pending = null;
    try {
      await fetch("/api/permissions/decide", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ scope, grant }),
      });
    } catch {
      // proxy unreachable — treat as denied
      return false;
    }
    return grant !== "deny";
  }

  /** Current grants as known by the backend (for the settings panel). */
  async grants(): Promise<Record<string, Grant>> {
    try {
      const res = await fetch("/api/permissions");
      if (res.ok) {
        const body = await res.json();
        return { ...body.persisted, ...body.session };
      }
    } catch {
      // proxy unreachable
    }
    return {};
  }

  /** Clears a grant server-side. */
  async revoke(scope: PermissionScope): Promise<void> {
    try {
      await fetch("/api/permissions/decide", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ scope, grant: "reset" }),
      });
    } catch {
      // proxy unreachable
    }
  }
}

export const permissions = new PermissionStore();
