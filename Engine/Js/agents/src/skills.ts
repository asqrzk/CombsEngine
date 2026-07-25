/**
 * Skills connector: loadable skill folders.
 *
 * A skill is a directory with:
 *   SKILL.md   — frontmatter (name, description) + the instruction body
 *   tools.ts   — optional module exporting `tools: Tool[]` (or a default export)
 *
 * Loading a skill injects its instruction body into the agent's system
 * prompt and registers its tools. This mirrors the agent-skills convention
 * (like this CLI's own skill system) so existing skill folders work as-is.
 */

import { getLogger } from "@combs/telemetry";
import type { Tool } from "./tools.ts";
import { ToolRegistry } from "./tools.ts";

const log = getLogger("combs.skills");

export interface Skill {
  name: string;
  description: string;
  /** The instruction body injected into the system prompt. */
  instructions: string;
  tools: Tool[];
  /** Source directory. */
  path: string;
}

function parseFrontmatter(markdown: string): { meta: Record<string, string>; body: string } {
  const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(markdown);
  if (!match) return { meta: {}, body: markdown };
  const meta: Record<string, string> = {};
  for (const line of match[1].split("\n")) {
    const idx = line.indexOf(":");
    if (idx > 0) meta[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
  }
  return { meta, body: match[2].trim() };
}

/** Loads one skill from a directory. */
export async function loadSkill(path: string): Promise<Skill> {
  const markdown = await Deno.readTextFile(`${path}/SKILL.md`);
  const { meta, body } = parseFrontmatter(markdown);

  let tools: Tool[] = [];
  try {
    const stat = await Deno.stat(`${path}/tools.ts`);
    if (stat.isFile) {
      const module = await import(`file://${await Deno.realPath(`${path}/tools.ts`)}`);
      const exported = module.tools ?? module.default ?? [];
      tools = Array.isArray(exported) ? exported : [exported];
    }
  } catch (e) {
    if (!(e instanceof Deno.errors.NotFound)) throw e;
  }

  const skill: Skill = {
    name: meta.name ?? path.split("/").filter(Boolean).pop() ?? "unnamed",
    description: meta.description ?? "",
    instructions: body,
    tools,
    path,
  };
  log.info("skill loaded", { name: skill.name, tools: skill.tools.length });
  return skill;
}

/** Loads every skill directory under `root` (one level deep). */
export async function loadSkills(root: string): Promise<Skill[]> {
  const skills: Skill[] = [];
  for await (const entry of Deno.readDir(root)) {
    if (!entry.isDirectory) continue;
    try {
      await Deno.stat(`${root}/${entry.name}/SKILL.md`);
      skills.push(await loadSkill(`${root}/${entry.name}`));
    } catch (e) {
      if (!(e instanceof Deno.errors.NotFound)) {
        log.warn("skill failed to load", { dir: entry.name, error: String(e) });
      }
    }
  }
  return skills;
}

/** Applies skills to an agent: registers tools, returns the prompt block. */
export function applySkills(skills: Skill[], registry: ToolRegistry): string {
  const blocks: string[] = [];
  for (const skill of skills) {
    registry.registerAll(skill.tools);
    blocks.push(`## Skill: ${skill.name}\n${skill.instructions}`);
  }
  return blocks.join("\n\n");
}
