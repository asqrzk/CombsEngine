/**
 * @combs-edge/combs-mesh — resolves the prebuilt CombsMesh native library
 * installed by install.js into vendor/.
 *
 * Resolution order for libPath(): vendor/ → COMBS_MESH_LIB env override →
 * null (callers get a helpful error message via requireLib()).
 */

const fs = require("node:fs");
const path = require("node:path");

const LIB_NAMES = {
  darwin: "libcombsmesh_ffi.dylib",
  linux: "libcombsmesh_ffi.so",
  win32: "combsmesh_ffi.dll",
};

function libFileName() {
  return LIB_NAMES[process.platform] || "libcombsmesh_ffi.so";
}

/** Absolute path to libcombsmesh_ffi, or null when not installed. */
function libPath() {
  const vendored = path.join(__dirname, "vendor", libFileName());
  if (fs.existsSync(vendored)) return vendored;
  const override = process.env.COMBS_MESH_LIB;
  if (override && fs.existsSync(override)) return override;
  return null;
}

/** Absolute path to combsmesh.h, or null when not installed. */
function headerPath() {
  const vendored = path.join(__dirname, "vendor", "combsmesh.h");
  return fs.existsSync(vendored) ? vendored : null;
}

/** libPath() or throw with build instructions. */
function requireLib() {
  const lib = libPath();
  if (!lib) {
    throw new Error(
      "[combs-mesh] native library not installed. Reinstall the package, " +
        "set COMBS_MESH_LIB=/path/to/libcombsmesh_ffi, or build from source: " +
        "cd Engine/Core && cargo build --release -p combs-mesh-ffi"
    );
  }
  return lib;
}

module.exports = { libPath, headerPath, requireLib };
