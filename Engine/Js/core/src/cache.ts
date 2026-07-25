/**
 * Model cache: downloads models from HuggingFace into a local store with
 * streaming progress, so multi-GB weights never pass through JS memory.
 *
 * Store layout: `<root>/<preset-id>/<file>`. Root defaults to
 * `$COMBS_HOME/models`, else `~/.cache/combs/models`.
 */
import type { ModelPreset } from "./types.ts";

const HF_BASE = "https://huggingface.co";

const REQUIRED_FILES = ["config.json", "tokenizer.json", "model.safetensors"];

export class ModelCache {
  constructor(readonly root: string = defaultStoreRoot()) {}

  /** Directory of a cached model. */
  modelDir(id: string): string {
    return `${this.root}/${id}`;
  }

  /** True when every required file is present locally. */
  async has(preset: ModelPreset): Promise<boolean> {
    try {
      for (const f of new Set([...REQUIRED_FILES, ...preset.files])) {
        const stat = await Deno.stat(`${this.modelDir(preset.id)}/${f}`);
        if (!stat.isFile || stat.size === 0) return false;
      }
      return true;
    } catch {
      return false;
    }
  }

  /** Downloads a preset's files (skipping existing ones). */
  async pull(
    preset: ModelPreset,
    onProgress?: (p: { file: string; received: number; total: number | null }) => void,
  ): Promise<string> {
    const dir = this.modelDir(preset.id);
    await Deno.mkdir(dir, { recursive: true });
    for (const file of preset.files) {
      const dest = `${dir}/${file}`;
      if (await exists(dest)) continue;
      await download(`${HF_BASE}/${preset.hfRepo}/resolve/main/${file}`, dest, (p) =>
        onProgress?.({ file, ...p }),
      );
    }
    return dir;
  }

  /** Removes a cached model. */
  async remove(id: string): Promise<void> {
    await Deno.remove(this.modelDir(id), { recursive: true });
  }
}

function defaultStoreRoot(): string {
  const combsHome = Deno.env.get("COMBS_HOME");
  if (combsHome) return `${combsHome}/models`;
  const home = Deno.env.get("HOME") ?? Deno.env.get("USERPROFILE") ?? ".";
  return `${home}/.cache/combs/models`;
}

async function exists(path: string): Promise<boolean> {
  try {
    return (await Deno.stat(path)).isFile;
  } catch {
    return false;
  }
}

/** Streams a URL to disk in chunks (constant memory). */
async function download(
  url: string,
  dest: string,
  onProgress?: (p: { received: number; total: number | null }) => void,
): Promise<void> {
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(`download failed: ${url} -> HTTP ${response.status}`);
  }
  const total = response.headers.get("content-length");
  const file = await Deno.open(dest, { write: true, create: true, truncate: true });
  try {
    let received = 0;
    for await (const chunk of response.body) {
      await file.write(chunk);
      received += chunk.length;
      onProgress?.({ received, total: total ? Number(total) : null });
    }
  } finally {
    file.close();
  }
}
