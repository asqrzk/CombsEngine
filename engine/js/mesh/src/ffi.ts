/**
 * Raw FFI bindings to libcombsmesh_ffi (Deno-only).
 *
 * Mirrors @combs/core's ffi.ts: library resolution order, borrowed/owned
 * C-string readback, NUL-terminated JSON buffers. All exports are
 * panic-fenced on the Rust side; errors are read back via
 * `combsmesh_last_error` (thread-local, borrowed).
 */

const SYMBOLS = {
  combsmesh_last_error: { parameters: [], result: "pointer" },
  combsmesh_string_free: { parameters: ["pointer"], result: "void" },
  combsmesh_bytes_free: { parameters: ["pointer", "usize"], result: "void" },
  combsmesh_init: { parameters: ["pointer", "usize"], result: "i32" },
  combsmesh_shutdown: { parameters: [], result: "i32" },
  combsmesh_encrypt_memory: {
    parameters: ["pointer", "usize", "pointer", "pointer"],
    result: "i32",
  },
  combsmesh_decrypt_memory: {
    parameters: ["pointer", "usize", "pointer", "pointer"],
    result: "i32",
  },
  combsmesh_render_sprite: {
    parameters: ["pointer", "usize", "u32", "pointer", "pointer"],
    result: "i32",
  },
  combsmesh_infer: { parameters: ["pointer", "pointer"], result: "i32" },
  combsmesh_op_json: { parameters: ["pointer"], result: "pointer" },
} as const;

export type MeshLib = Deno.DynamicLibrary<typeof SYMBOLS>;

function libFileName(): string {
  switch (Deno.build.os) {
    case "darwin":
      return "libcombsmesh_ffi.dylib";
    case "windows":
      return "combsmesh_ffi.dll";
    default:
      return "libcombsmesh_ffi.so";
  }
}

/** Candidate locations for the native library, in priority order. */
export function libraryCandidates(): string[] {
  const here = new URL(".", import.meta.url).pathname;
  const core = `${here}../../../Core`;
  const name = libFileName();
  const platformDir = (() => {
    const arch = Deno.build.arch === "aarch64" ? "arm64" : "x86_64";
    switch (Deno.build.os) {
      case "darwin":
        return `macos-${arch}`;
      case "linux":
        return `linux-${arch}`;
      case "windows":
        return `windows-${arch}`;
      default:
        return `${Deno.build.os}-${arch}`;
    }
  })();
  return [
    `${core}/dist/${platformDir}/${name}`,
    `${core}/target/release/${name}`,
  ];
}

/**
 * Opens the native library. Resolution order: explicit argument →
 * COMBS_MESH_LIB env → COMBS_LIB env → Engine/Core/dist/<plat> →
 * Engine/Core/target/release.
 */
export function openLibrary(libraryPath?: string): MeshLib {
  const explicit = libraryPath ??
    Deno.env.get("COMBS_MESH_LIB") ??
    Deno.env.get("COMBS_LIB");
  const candidates = explicit ? [explicit] : libraryCandidates();
  const errors: string[] = [];
  for (const path of candidates) {
    try {
      return Deno.dlopen(path, SYMBOLS);
    } catch (e) {
      errors.push(`${path}: ${e}`);
    }
  }
  throw new Error(
    `could not load the combsmesh native library. Tried:\n  ${errors.join("\n  ")}\n` +
      `Build it with \`cargo build --release -p combs-mesh-ffi\` or \`cargo xtask bundle\`, ` +
      `or set COMBS_MESH_LIB.`,
  );
}

/** Reads a borrowed C string from the library (never freed). */
export function readBorrowedCString(ptr: Deno.PointerValue): string {
  if (!ptr) return "";
  return new Deno.UnsafePointerView(ptr).getCString();
}

/** Reads an owned C string and frees it with combsmesh_string_free. */
export function readOwnedCString(lib: MeshLib, ptr: Deno.PointerValue): string {
  if (!ptr) return "";
  const s = new Deno.UnsafePointerView(ptr).getCString();
  lib.symbols.combsmesh_string_free(ptr);
  return s;
}

/** Thread-local last-error string from the library. */
export function lastError(lib: MeshLib): string {
  return readBorrowedCString(lib.symbols.combsmesh_last_error());
}

const encoder = new TextEncoder();

/** NUL-terminated UTF-8 buffer for passing strings to the library. */
export function cstring(text: string): Uint8Array {
  const bytes = encoder.encode(text);
  const out = new Uint8Array(bytes.length + 1);
  out.set(bytes);
  return out;
}

/**
 * Reads an owned byte buffer produced by an out-param call and frees it
 * with combsmesh_bytes_free. `outPtr`/`outLen` are the 8-byte slots that
 * were passed to the library.
 */
export function readOwnedBytes(
  lib: MeshLib,
  outPtr: BigUint64Array,
  outLen: BigUint64Array,
): Uint8Array {
  const ptr = Deno.UnsafePointer.create(outPtr[0]);
  const len = Number(outLen[0]);
  if (!ptr || len === 0) return new Uint8Array(0);
  const bytes = new Uint8Array(len);
  new Deno.UnsafePointerView(ptr).copyInto(bytes);
  lib.symbols.combsmesh_bytes_free(ptr, BigInt(len));
  return bytes;
}

/** Fresh out-param slots for the `uint8_t** out, size_t* out_len` pattern. */
export function outSlots(): { outPtr: BigUint64Array; outLen: BigUint64Array } {
  return { outPtr: new BigUint64Array(1), outLen: new BigUint64Array(1) };
}
