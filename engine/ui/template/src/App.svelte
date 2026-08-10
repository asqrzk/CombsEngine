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
  import OrchestrationView from "./lib/components/OrchestrationView.svelte";
  import KvCacheView from "./lib/components/KvCacheView.svelte";
  import DebateKvView from "./lib/components/DebateKvView.svelte";
  import TowerView from "./lib/components/tower/TowerView.svelte";

  let config = $state<UiConfig | null>(null);
  let booted = $state(false);
  let hash = $state("");

  onMount(async () => {
    config = await loadConfig();
    themeStore.init(config.theme);
    monitor.start();
    if (config.authentication) {
      await authStore.init();
    }
    booted = true;
  });

  onMount(() => {
    hash = location.hash;
    const onHash = () => (hash = location.hash);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  });

  const needsAuth = $derived(
    booted && config?.authentication === true &&
    (!authStore.identity || !authStore.identity.backedUp ||
      (!authStore.passkeyRegistered && !authStore.passkeySkipped)),
  );
  const towerCapable = $derived(
    config?.mode === "multi-turn-ui" ||
    config?.mode === "orchestration-observe-ui" ||
    config?.mode === "kv-cache-ui" ||
    config?.mode === "debate-kv-ui",
  );
  const showTowerPage = $derived(towerCapable && hash === "#tower");
</script>

{#if !booted || !config}
  <div class="flex h-full items-center justify-center text-[rgb(var(--muted))]">
    <div class="animate-pulse text-sm">booting combs…</div>
  </div>
{:else}
  <div class="flex h-full flex-col">
    <TopBar mode={config.mode} {hash} />
    <main class="min-h-0 flex-1">
      {#if needsAuth}
        <AuthSetup />
      {:else}
        <!-- Keep-alive: both panes stay MOUNTED (visibility toggled only), so
             navigating to the tower and back never loses conversation state
             or kills a running orchestration loop. -->
        <div class="h-full" class:hidden={showTowerPage}>
          {#if config.mode === "debate-ui"}
            <DebateView {config} />
          {:else if config.mode === "roleplay-ui"}
            <RoleplayView {config} />
          {:else if config.mode === "multi-turn-ui"}
            <MultiTurnView {config} />
          {:else if config.mode === "orchestration-observe-ui"}
            <OrchestrationView {config} />
          {:else if config.mode === "kv-cache-ui"}
            <KvCacheView {config} />
          {:else if config.mode === "debate-kv-ui"}
            <DebateKvView {config} />
          {:else}
            <ChatView {config} />
          {/if}
        </div>
        {#if towerCapable}
          <div class="h-full" class:hidden={!showTowerPage}>
            <TowerView />
          </div>
        {/if}
      {/if}
    </main>
  </div>
  <PermissionDialog />
{/if}
