/**
 * Fine-grained permissions: every sensitive action (network downloads,
 * local caching) asks first — allow once / allow this session / allow
 * always / deny. Grants are persisted per scope in localStorage; session
 * grants live in memory only.
 */

export type Grant = "once" | "session" | "always" | "deny";

export type PermissionScope =
  | "network:download" // downloading models over the internet
  | "network:inference" // talking to the inference server
  | "storage:cache" // caching models/chats on this device
  | "storage:chats"; // persisting chat sessions

interface PendingRequest {
  scope: PermissionScope;
  detail: string;
  resolve: (grant: Grant) => void;
}

const KEY = "combs.permissions";

class PermissionStore {
  /** Pending permission request shown in the dialog (reactive). */
  pending = $state<PendingRequest | null>(null);

  private persisted: Record<string, Grant> = JSON.parse(localStorage.getItem(KEY) ?? "{}");
  private session = new Map<PermissionScope, Grant>();
  private onceUsed = new Set<PermissionScope>();

  /** Checks a scope; shows the dialog when no grant exists. Resolves true
   * when the action is allowed this time. */
  async require(scope: PermissionScope, detail: string): Promise<boolean> {
    const always = this.persisted[scope];
    if (always === "always") return true;
    if (always === "deny") return false;

    const sessionGrant = this.session.get(scope);
    if (sessionGrant === "session") return true;
    if (sessionGrant === "deny") return false;

    if (this.onceUsed.has(scope)) {
      // "allow once" was already consumed — ask again.
      this.onceUsed.delete(scope);
    }

    const grant = await new Promise<Grant>((resolve) => {
      this.pending = { scope, detail, resolve };
    });
    this.pending = null;

    switch (grant) {
      case "always":
        this.persisted[scope] = grant;
        localStorage.setItem(KEY, JSON.stringify(this.persisted));
        return true;
      case "session":
        this.session.set(scope, grant);
        return true;
      case "once":
        this.onceUsed.add(scope);
        return true;
      case "deny":
      default:
        this.session.set(scope, "deny");
        return false;
    }
  }

  /** Current persisted grants (for the settings panel). */
  grants(): Record<string, Grant> {
    return { ...this.persisted };
  }

  revoke(scope: PermissionScope): void {
    delete this.persisted[scope];
    this.session.delete(scope);
    localStorage.setItem(KEY, JSON.stringify(this.persisted));
  }
}

export const permissions = new PermissionStore();
