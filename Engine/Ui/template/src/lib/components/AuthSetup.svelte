<script lang="ts">
  import { authStore } from "../auth.svelte";
  import { registerPasskey, webauthnSupported } from "../passkey";
  import Button from "./ui/Button.svelte";
  import Card from "./ui/Card.svelte";

  let downloading = $state(false);
  let passkeyBusy = $state(false);
  let passkeyError = $state<string | null>(null);

  async function generate() {
    await authStore.generate();
  }

  async function backup() {
    downloading = true;
    await authStore.downloadBackup();
    downloading = false;
  }

  async function createPasskey() {
    passkeyBusy = true;
    passkeyError = null;
    const ok = await registerPasskey();
    passkeyBusy = false;
    if (ok) {
      authStore.passkeyRegistered = true;
    } else {
      passkeyError = "passkey creation failed or was cancelled";
    }
  }

  function skipPasskey() {
    // WebAuthn unsupported on this browser: degrade to click-approval.
    authStore.passkeyRegistered = false;
    authStore.passkeySkipped = true;
  }

  const needsBackup = $derived(authStore.identity !== null && !authStore.identity.backedUp);
  const needsPasskey = $derived(
    authStore.identity !== null &&
      authStore.identity.backedUp &&
      !authStore.passkeyRegistered &&
      !authStore.passkeySkipped,
  );
</script>

<div class="mx-auto flex max-w-md flex-col gap-4 pt-16">
  <Card>
    {#if !authStore.identity}
      <!-- first run: no identity on this device -->
      <h1 class="text-xl font-semibold">Create your identity</h1>
      <p class="mt-2 text-sm text-[rgb(var(--muted))]">
        Combs generates a private/public keypair on this device. Your key signs
        saved sessions and proves this install is yours. After you download the
        backup, the private key is stored only as a non-extractable device key.
      </p>
      <div class="mt-4">
        <Button variant="accent" onclick={generate}>Generate keypair</Button>
      </div>
    {:else}
      <!-- identity already exists: load it and finish any remaining steps -->
      <h1 class="text-xl font-semibold">
        {needsBackup || needsPasskey ? "Finish setting up" : "Welcome back"}
      </h1>
      <p class="mt-2 text-sm text-[rgb(var(--muted))]">
        Your identity was loaded from this device.
      </p>

      <div class="mt-4 rounded-lg border bg-[rgb(var(--bg))] p-3">
        <div class="text-xs text-[rgb(var(--muted))]">public key fingerprint</div>
        <code class="text-sm break-all">{authStore.identity.fingerprint}</code>
      </div>

      {#if needsBackup}
        <p class="mt-3 text-sm text-amber-600 dark:text-amber-400">
          ⚠ Back up your private key now. Without it, your saved data cannot
          be recovered. After the backup, the key is sealed as a
          non-extractable device key.
        </p>
        <div class="mt-3">
          <Button variant="accent" onclick={backup} disabled={downloading}>
            Download private key backup
          </Button>
        </div>
      {:else if needsPasskey}
        <p class="mt-3 text-sm text-green-600 dark:text-green-400">✓ Key backed up.</p>
        <p class="mt-3 text-sm text-[rgb(var(--muted))]">
          {#if webauthnSupported}
            Finally, create a <strong>device passkey</strong> (Touch ID / Windows
            Hello / security key). Every permission approval — model downloads,
            agents, internet access — is confirmed with it.
          {:else}
            This browser does not support WebAuthn passkeys. Permission
            approvals will fall back to click-confirmation.
          {/if}
        </p>
        {#if passkeyError}
          <p class="mt-2 text-xs text-red-600 dark:text-red-400">{passkeyError}</p>
        {/if}
        <div class="mt-3 flex gap-2">
          {#if webauthnSupported}
            <Button variant="accent" onclick={createPasskey} disabled={passkeyBusy}>
              {passkeyBusy ? "waiting for device…" : "Create passkey"}
            </Button>
          {:else}
            <Button variant="ghost" onclick={skipPasskey}>Continue without passkey</Button>
          {/if}
        </div>
      {:else}
        <p class="mt-3 text-sm text-green-600 dark:text-green-400">
          ✓ Identity loaded{authStore.passkeyRegistered ? ", passkey active" : ""} — you're all set.
        </p>
      {/if}
    {/if}
  </Card>
</div>
