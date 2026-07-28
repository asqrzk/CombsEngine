<script lang="ts">
  import { onMount } from "svelte";
  import { loadConfig, type UiConfig } from "./lib/config";
  import { themeStore } from "./lib/theme.svelte";
  import { authStore } from "./lib/auth.svelte";
  import { monitor } from "./lib/monitor.svelte";
  import TopBar from "./lib/components/TopBar.svelte";
  import AuthSetup from "./lib/components/AuthSetup.svelte";
  import PermissionDialog from "./lib/components/PermissionDialog.svelte";
  import ChatView from "./lib/components/ChatView.svelte";
  import DebateView from "./lib/components/DebateView.svelte";
  import RoleplayView from "./lib/components/RoleplayView.svelte";
  import MultiTurnView from "./lib/components/MultiTurnView.svelte";

  let config = $state<UiConfig | null>(null);
  let booted = $state(false);

  onMount(async () => {
    config = await loadConfig();
    themeStore.init(config.theme);
    monitor.start();
    if (config.authentication) {
      await authStore.init();
    }
    booted = true;
  });

  const needsAuth = $derived(
    booted && config?.authentication === true &&
    (!authStore.identity || !authStore.identity.backedUp ||
      (!authStore.passkeyRegistered && !authStore.passkeySkipped)),
  );
</script>

{#if !booted || !config}
  <div class="flex h-full items-center justify-center text-[rgb(var(--muted))]">
    <div class="animate-pulse text-sm">booting combs…</div>
  </div>
{:else}
  <div class="flex h-full flex-col">
    <TopBar />
    <main class="min-h-0 flex-1">
      {#if needsAuth}
        <AuthSetup />
      {:else if config.mode === "debate-ui"}
        <DebateView {config} />
      {:else if config.mode === "roleplay-ui"}
        <RoleplayView {config} />
      {:else if config.mode === "multi-turn-ui"}
        <MultiTurnView {config} />
      {:else}
        <ChatView {config} />
      {/if}
    </main>
  </div>
  <PermissionDialog />
{/if}
