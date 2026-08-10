<script lang="ts">
  import { permissions } from "../permissions.svelte";
  import { authStore } from "../auth.svelte";
  import { approveWithPasskey, registerPasskey } from "../passkey";
  import Button from "./ui/Button.svelte";
  import Dialog from "./ui/Dialog.svelte";

  const SCOPE_LABELS: Record<string, string> = {
    "network:download": "download models over the internet",
    "network:inference": "connect to the inference server",
    "network:internet": "give an agent internet access (sandboxed)",
    "system:subprocess": "start a local engine/agent on this device",
    "storage:cache": "cache data on this device",
    "storage:chats": "save chats on this device",
  };

  let busy = $state(false);
  let error = $state<string | null>(null);
  /** True when the passkey assertion failed and re-registration is offered. */
  let needsReregister = $state(false);

  async function choose(grant: "once" | "session" | "always" | "deny") {
    // Deny never needs a passkey. Allow-grants do — when a passkey is
    // registered, the proxy only records the grant after a verified
    // assertion. Without a registered passkey (unsupported browser),
    // degrade to click-approval.
    if (grant !== "deny" && authStore.passkeyRegistered) {
      busy = true;
      error = null;
      const ok = await approveWithPasskey();
      busy = false;
      if (!ok) {
        // The passkey the proxy knows about is no longer usable on this
        // device (deleted, or the store was reset). Offer re-registration
        // instead of looping on a dead credential.
        needsReregister = true;
        error = "passkey approval failed — your device passkey may have changed";
        return;
      }
    }
    permissions.pending?.resolve(grant);
  }

  async function reregister() {
    busy = true;
    error = null;
    const ok = await registerPasskey();
    busy = false;
    if (ok) {
      authStore.passkeyRegistered = true;
      needsReregister = false;
      error = null;
      // The original permission request is still pending — the user can
      // approve it now with the fresh passkey.
    } else {
      error = "passkey re-registration failed or was cancelled";
    }
  }
</script>

{#if permissions.pending}
  <Dialog open title="Permission requested">
    <p class="text-sm">
      Combs wants to
      <strong>{SCOPE_LABELS[permissions.pending.scope] ?? permissions.pending.scope}</strong>.
    </p>
    <p class="mt-1 text-sm text-[rgb(var(--muted))]">{permissions.pending.detail}</p>

    {#if authStore.passkeyRegistered}
      <p class="mt-2 text-xs text-[rgb(var(--muted))]">🔐 approvals require your device passkey</p>
    {/if}
    {#if error}
      <p class="mt-2 text-xs text-red-600 dark:text-red-400">{error}</p>
    {/if}

    {#if needsReregister}
      <div class="mt-3 rounded-lg border border-amber-400 bg-amber-50 p-3 dark:bg-amber-950/30">
        <p class="text-xs text-amber-700 dark:text-amber-300">
          Re-create your device passkey, then approve again.
        </p>
        <div class="mt-2">
          <Button variant="accent" disabled={busy} onclick={reregister}>
            {busy ? "waiting for device…" : "Re-create passkey"}
          </Button>
        </div>
      </div>
    {/if}

    <div class="mt-4 grid grid-cols-2 gap-2">
      <Button variant="ghost" disabled={busy} onclick={() => choose("once")}>Allow once</Button>
      <Button variant="ghost" disabled={busy} onclick={() => choose("session")}>Allow this session</Button>
      <Button variant="accent" disabled={busy} onclick={() => choose("always")}>Allow always</Button>
      <Button variant="danger" disabled={busy} onclick={() => choose("deny")}>Deny</Button>
    </div>
  </Dialog>
{/if}
