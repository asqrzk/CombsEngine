#!/usr/bin/env node
/**
 * postinstall — downloads the prebuilt `combs` binary for this platform from
 * the GitHub Release that matches this package version, into vendor/.
 *
 * Asset names (produced by .github/workflows/release.yml):
 *   combs-<version>-macos-arm64.tar.gz
 *   combs-<version>-macos-x86_64.tar.gz
 *   combs-<version>-linux-x86_64.tar.gz
 *   combs-<version>-windows-x86_64.zip
 *
 * Override the release repo with COMBS_RELEASE_REPO=owner/name.
 * Skip the download with COMBS_SKIP_BINARY_DOWNLOAD=1 (e.g. when `combs`
 * is already on PATH or built from source).
 */

const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const https = require("node:https");
const path = require("node:path");

const PKG = require("./package.json");
const REPO = process.env.COMBS_RELEASE_REPO || "asqrzk/CombsEngine";
const VENDOR = path.join(__dirname, "vendor");

const PLATFORMS = { darwin: "macos", linux: "linux", win32: "windows" };
const ARCHES = { arm64: "arm64", x64: "x86_64" };

function assetName() {
  const platform = PLATFORMS[process.platform];
  const arch = ARCHES[process.arch];
  if (!platform || !arch) {
    throw new Error(`unsupported platform: ${process.platform}/${process.arch}`);
  }
  const ext = platform === "windows" ? "zip" : "tar.gz";
  return `combs-${PKG.version}-${platform}-${arch}.${ext}`;
}

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https
      .get(url, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          file.close();
          fs.unlinkSync(dest);
          return resolve(download(res.headers.location, dest));
        }
        if (res.statusCode !== 200) {
          file.close();
          fs.unlinkSync(dest);
          return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        }
        res.pipe(file);
        file.on("finish", () => file.close(resolve));
      })
      .on("error", reject);
  });
}

async function main() {
  if (process.env.COMBS_SKIP_BINARY_DOWNLOAD === "1") {
    console.log("[combs-engine] COMBS_SKIP_BINARY_DOWNLOAD=1 — skipping binary download");
    return;
  }
  const asset = assetName();
  const url = `https://github.com/${REPO}/releases/download/v${PKG.version}/${asset}`;
  const tmp = path.join(VENDOR, asset);

  fs.rmSync(VENDOR, { recursive: true, force: true });
  fs.mkdirSync(VENDOR, { recursive: true });

  console.log(`[combs-engine] downloading ${asset}`);
  try {
    await download(url, tmp);
  } catch (e) {
    console.warn(`[combs-engine] download failed: ${e.message}`);
    console.warn("[combs-engine] no prebuilt binary for this platform/version yet.");
    console.warn("[combs-engine] build from source instead:");
    console.warn("[combs-engine]   cargo install --path Engine/Core/combs-cli");
    process.exit(0); // don't hard-fail the whole npm install
  }

  const inner = asset.replace(/\.(tar\.gz|zip)$/, "");
  if (asset.endsWith(".zip")) {
    execFileSync("tar", ["-xf", tmp, "-C", VENDOR]); // bsdtar on win10+ handles zip
  } else {
    execFileSync("tar", ["-xzf", tmp, "-C", VENDOR]);
  }
  fs.unlinkSync(tmp);

  const binName = process.platform === "win32" ? "combs.exe" : "combs";
  const binPath = path.join(VENDOR, inner, binName);
  fs.chmodSync(binPath, 0o755);
  console.log(`[combs-engine] installed combs ${PKG.version} -> ${binPath}`);
}

main().catch((e) => {
  console.warn(`[combs-engine] postinstall error: ${e.message}`);
  process.exit(0);
});
