#!/usr/bin/env node
/** `combs` shim — forwards all args to the platform binary in vendor/. */

const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const binName = process.platform === "win32" ? "combs.exe" : "combs";
const vendor = path.join(__dirname, "..", "vendor");

function findBinary() {
  if (!fs.existsSync(vendor)) return null;
  // vendor/combs-<version>-<platform>-<arch>/combs
  for (const entry of fs.readdirSync(vendor)) {
    const candidate = path.join(vendor, entry, binName);
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

const bin = process.env.COMBS_BIN || findBinary();
if (!bin) {
  console.error(
    "[combs-engine] binary not installed. Reinstall, or set COMBS_BIN=/path/to/combs,\n" +
      "or build from source: cargo install --path Engine/Core/combs-cli"
  );
  process.exit(1);
}

const res = spawnSync(bin, process.argv.slice(2), { stdio: "inherit" });
process.exit(res.status ?? 1);
