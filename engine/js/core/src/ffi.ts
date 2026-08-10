/**
 * Raw FFI bindings to libcombs_ffi (Deno-only).
 *
 * `combs_chat_completion` blocks the calling thread and invokes the stream
 * callback on it; it is declared `nonblocking: true` so Deno runs it on a
 * blocking thread, and callers must use `Deno.UnsafeCallback.threadSafe`
 * for the callback.
 */
import type { DeviceCaps } from "./types.ts";

const SYMBOLS = {
  combs_device_caps_json: { parameters: [], result: "pointer" },
  combs_engine_create: { parameters: ["pointer"], result: "pointer" },
  combs_engine_destroy: { parameters: ["pointer"], result: "void" },
  combs_engine_metadata_json: { parameters: ["pointer"], result: "pointer" },
  combs_chat_completion: {
    parameters: ["pointer", "pointer", "pointer", "pointer", "pointer"],
    result: "i32",
    nonblocking: true,
  },
  combs_cancel: { parameters: ["pointer"], result: "i32" },
  combs_last_error: { parameters: [], result: "pointer" },
  combs_string_free: { parameters: ["pointer"], result: "void" },
} as const;

export type CombsLib = Deno.DynamicLibrary<typeof SYMBOLS>;

function libFileName(): string {
  switch (Deno.build.os) {
    case "darwin":
      return "libcombs_ffi.dylib";
    case "windows":
      return "combs_ffi.dll";
    default:
      return "libcombs_ffi.so";
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

/** Opens the native library. Honors COMBS_LIB for an explicit path. */
export function openLibrary(libraryPath?: string): CombsLib {
  const candidates = libraryPath
    ? [libraryPath]
    : Deno.env.get("COMBS_LIB")
    ? [Deno.env.get("COMBS_LIB")!]
    : libraryCandidates();
  const errors: string[] = [];
  for (const path of candidates) {
    try {
      return Deno.dlopen(path, SYMBOLS);
    } catch (e) {
      errors.push(`${path}: ${e}`);
    }
  }
  throw new Error(
    `could not load the combs native library. Tried:\n  ${errors.join("\n  ")}\n` +
      `Build it with \`cargo xtask build --release\` or \`cargo xtask bundle\`, ` +
      `or set COMBS_LIB.`,
  );
}

/** Reads a borrowed C string from the library (never freed). */
export function readBorrowedCString(ptr: Deno.PointerValue): string {
  if (!ptr) return "";
  return new Deno.UnsafePointerView(ptr).getCString();
}

/** Reads an owned C string and frees it with combs_string_free. */
export function readOwnedCString(lib: CombsLib, ptr: Deno.PointerValue): string {
  if (!ptr) return "";
  const s = new Deno.UnsafePointerView(ptr).getCString();
  lib.symbols.combs_string_free(ptr);
  return s;
}

/** Thread-local last-error string from the library. */
export function lastError(lib: CombsLib): string {
  return readBorrowedCString(lib.symbols.combs_last_error());
}

const encoder = new TextEncoder();

/** NUL-terminated UTF-8 buffer for passing JSON to the library. */
export function cstring(text: string): Uint8Array {
  const bytes = encoder.encode(text);
  const out = new Uint8Array(bytes.length + 1);
  out.set(bytes);
  return out;
}

/** Queries device capabilities (for the DevicePlanner). */
export function deviceCaps(lib: CombsLib): DeviceCaps {
  const ptr = lib.symbols.combs_device_caps_json();
  if (!ptr) throw new Error(`combs_device_caps_json failed: ${lastError(lib)}`);
  return JSON.parse(readOwnedCString(lib, ptr)) as DeviceCaps;
}
