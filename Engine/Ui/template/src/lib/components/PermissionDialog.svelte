<script lang="ts">
  import { permissions } from "../permissions";
  import Button from "./ui/Button.svelte";
  import Dialog from "./ui/Dialog.svelte";

  const SCOPE_LABELS: Record<string, string> = {
    "network:download": "download models over the internet",
    "network:inference": "connect to the inference server",
    "storage:cache": "cache models on this device",
    "storage:chats": "save chats on this device",
  };

  function choose(grant: "once" | "session" | "always" | "deny") {
    permissions.pending?.resolve(grant);
  }
</script>

{#if permissions.pending}
  <Dialog open title="Permission requested">
    <p class="text-sm">
      Combs wants to
      <strong>{SCOPE_LABELS[permissions.pending.scope] ?? permissions.pending.scope}</strong>.
    </p>
    <p class="mt-1 text-sm text-[rgb(var(--muted))]">{permissions.pending.detail}</p>

    <div class="mt-4 grid grid-cols-2 gap-2">
      <Button variant="ghost" onclick={() => choose("once")}>Allow once</Button>
      <Button variant="ghost" onclick={() => choose("session")}>Allow this session</Button>
      <Button variant="accent" onclick={() => choose("always")}>Allow always</Button>
      <Button variant="danger" onclick={() => choose("deny")}>Deny</Button>
    </div>
  </Dialog>
{/if}
