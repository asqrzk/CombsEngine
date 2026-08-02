#!/usr/bin/env node
/**
 * postinstall — downloads the GitHub Release asset matching this package
 * version (the SAME asset as combs-engine — mesh ships inside it) and
 * extracts only libcombsmesh_ffi.* + combsmesh.h into vendor/.
 *
 * Asset names (produced by .github/workflows/release.yml):
 *   combs-<version>-macos-arm64.tar.gz
 *   combs-<version>-macos-x86_64.tar.gz
 *   combs-<version>-linux-x86_64.tar.gz
 *   combs-<version>-windows-x86_64.zip
 *
 * Override the release repo with COMBS_RELEASE_REPO=owner/name.
 * Skip the download with COMBS_SKIP_BINARY_DOWNLOAD=1.
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

/** Library + header file names kept from the asset, per platform. */
function keepNames() {
  if (process.platform === "win32") return ["combsmesh_ffi.dll", "combsmesh.h"];
  if (process.platform === "darwin") return ["libcombsmesh_ffi.dylib", "libcombsmesh_ffi.a", "combsmesh.h"];
  return ["libcombsmesh_ffi.so", "libcombsmesh_ffi.a", "combsmesh.h"];
}

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
    console.log("[combs-mesh] COMBS_SKIP_BINARY_DOWNLOAD=1 — skipping library download");
    return;
  }
  const asset = assetName();
  const url = `https://github.com/${REPO}/releases/download/v${PKG.version}/${asset}`;
  const tmp = path.join(VENDOR, asset);

  fs.rmSync(VENDOR, { recursive: true, force: true });
  fs.mkdirSync(VENDOR, { recursive: true });

  console.log(`[combs-mesh] downloading ${asset}`);
  try {
    await download(url, tmp);
  } catch (e) {
    console.warn(`[combs-mesh] download failed: ${e.message}`);
    console.warn("[combs-mesh] no prebuilt library for this platform/version yet.");
    console.warn("[combs-mesh] build from source instead:");
    console.warn("[combs-mesh]   cd Engine/Core && cargo build --release -p combs-mesh-ffi");
    process.exit(0); // don't hard-fail the whole npm install
  }

  // Extract into a scratch dir, then keep only the mesh artifacts flat in
  // vendor/ (the asset also carries the `combs` CLI + libcombs_ffi).
  const scratch = path.join(VENDOR, "stage");
  fs.mkdirSync(scratch, { recursive: true });
  if (asset.endsWith(".zip")) {
    execFileSync("tar", ["-xf", tmp, "-C", scratch]); // bsdtar on win10+ handles zip
  } else {
    execFileSync("tar", ["-xzf", tmp, "-C", scratch]);
  }
  fs.unlinkSync(tmp);

  const keep = keepNames();
  const inner = path.join(scratch, asset.replace(/\.(tar\.gz|zip)$/, ""));
  for (const name of keep) {
    const src = path.join(inner, name);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(VENDOR, name));
  }
  fs.rmSync(scratch, { recursive: true, force: true });
  console.log(`[combs-mesh] installed libcombsmesh_ffi ${PKG.version} -> ${VENDOR}`);
}

main().catch((e) => {
  console.warn(`[combs-mesh] postinstall error: ${e.message}`);
  process.exit(0);
});
