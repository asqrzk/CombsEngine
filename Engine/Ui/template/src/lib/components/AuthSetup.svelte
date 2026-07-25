<script lang="ts">
  import { authStore } from "../auth";
  import Button from "./ui/Button.svelte";
  import Card from "./ui/Card.svelte";

  let downloading = $state(false);

  async function generate() {
    await authStore.generate();
  }

  async function backup() {
    downloading = true;
    await authStore.downloadBackup();
    downloading = false;
  }
</script>

<div class="mx-auto flex max-w-md flex-col gap-4 pt-16">
  <Card>
    <h1 class="text-xl font-semibold">Create your identity</h1>
    <p class="mt-2 text-sm text-[rgb(var(--muted))]">
      Combs generates a private/public keypair on this device. Your key signs
      saved sessions and proves this install is yours.
    </p>

    {#if !authStore.identity}
      <div class="mt-4">
        <Button variant="accent" onclick={generate}>Generate keypair</Button>
      </div>
    {:else}
      <div class="mt-4 rounded-lg border bg-[rgb(var(--bg))] p-3">
        <div class="text-xs text-[rgb(var(--muted))]">public key fingerprint</div>
        <code class="text-sm break-all">{authStore.identity.fingerprint}</code>
      </div>

      {#if !authStore.identity.backedUp}
        <p class="mt-3 text-sm text-amber-600 dark:text-amber-400">
          ⚠ Back up your private key now. Without it, your saved data cannot
          be recovered.
        </p>
        <div class="mt-3">
          <Button variant="accent" onclick={backup} disabled={downloading}>
            Download private key backup
          </Button>
        </div>
      {:else}
        <p class="mt-3 text-sm text-green-600 dark:text-green-400">✓ Backup downloaded — you're all set.</p>
      {/if}
    {/if}
  </Card>
</div>
