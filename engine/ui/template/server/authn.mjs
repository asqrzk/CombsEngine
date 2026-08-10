/**
 * Passkey (WebAuthn) ceremonies for the proxy, via SimpleWebAuthn.
 *
 * The passkey is how the user APPROVES permission requests: an allow
 * decision in the dialog triggers an authentication ceremony, and the
 * grant is only recorded after the proxy verifies the assertion here.
 *
 * Endpoints (wired in proxy.mjs under /api/auth/passkey/):
 *   GET  status            → {registered}
 *   POST register-options  {username?} → PublicKeyCredentialCreationOptions
 *   POST register-verify   {response}  → {verified}
 *   POST auth-options      {}          → PublicKeyCredentialRequestOptions
 *   POST auth-verify       {response}  → {verified}
 *
 * Credentials persist GLOBALLY in $COMBS_HOME/authn.json (default
 * ~/.cache/combs/authn.json) — one device passkey serves every chew app
 * and every port, so users create it once and are never asked again.
 * (RP ID is "localhost" — port-independent, so the same credential works
 * on any local origin.) Challenges are single-slot with a 5-minute TTL.
 * RP_ID/origins configurable via COMBS_RP_ID / COMBS_ORIGINS.
 */

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  generateAuthenticationOptions,
  generateRegistrationOptions,
  verifyAuthenticationResponse,
  verifyRegistrationResponse,
} from "@simplewebauthn/server";
import { b64decode, b64encode } from "@combs-edge/combs-zerotrust";

const HERE = path.dirname(fileURLToPath(import.meta.url));
// Global credential store: shared across all chew apps/ports on this
// machine (unlike per-app permissions.json).
const CRED_DIR = process.env.COMBS_HOME ||
  path.join(process.env.HOME || process.env.USERPROFILE || HERE, ".cache", "combs");
const CRED_FILE = path.join(CRED_DIR, "authn.json");

const RP_NAME = "Combs UI";
const RP_ID = process.env.COMBS_RP_ID || "localhost";
const ORIGINS = (process.env.COMBS_ORIGINS || "http://localhost:5173,http://localhost:8787")
  .split(",")
  .map((s) => s.trim());
const CHALLENGE_TTL = 5 * 60 * 1000;

let credentials = loadCredentials();

function loadCredentials() {
  try {
    return JSON.parse(fs.readFileSync(CRED_FILE, "utf8")).credentials ?? [];
  } catch { /* first run */ }
  return [];
}

// The credential file is global (~/.cache/combs/authn.json) and shared by
// every proxy on this machine — another app may register or the user may
// delete/reset it while this proxy is running. Re-read before status/auth
// so a deleted store doesn't leave a phantom in-memory credential that
// makes every assertion fail with "unknown credential".
function refreshCredentials() {
  credentials = loadCredentials();
}

const challenges = new Map(); // "reg" | "auth" -> {challenge, expires}

function setChallenge(kind, challenge) {
  challenges.set(kind, { challenge, expires: Date.now() + CHALLENGE_TTL });
}
function takeChallenge(kind) {
  const c = challenges.get(kind);
  challenges.delete(kind);
  if (!c || c.expires < Date.now()) throw new Error("challenge expired or missing");
  return c.challenge;
}

async function saveCredentials() {
  await fsp.mkdir(CRED_DIR, { recursive: true });
  await fsp.writeFile(CRED_FILE, JSON.stringify({ credentials }, null, 2));
}

export async function handleAuthn(req, res, url, readBody, send) {
  const action = url.pathname.slice("/api/auth/passkey/".length);

  if (action === "status" && req.method === "GET") {
    refreshCredentials();
    return send(res, 200, { registered: credentials.length > 0 });
  }

  if (action === "register-options" && req.method === "POST") {
    const { username } = JSON.parse((await readBody(req)).toString("utf8") || "{}");
    const options = await generateRegistrationOptions({
      rpName: RP_NAME,
      rpID: RP_ID,
      userName: username || "combs-user",
      attestationType: "none",
      excludeCredentials: credentials.map((c) => ({ id: c.id, transports: c.transports })),
      authenticatorSelection: { residentKey: "preferred", userVerification: "preferred" },
    });
    setChallenge("reg", options.challenge);
    return send(res, 200, options);
  }

  if (action === "register-verify" && req.method === "POST") {
    const { response } = JSON.parse((await readBody(req)).toString("utf8") || "{}");
    const verification = await verifyRegistrationResponse({
      response,
      expectedChallenge: takeChallenge("reg"),
      expectedOrigin: ORIGINS,
      expectedRPID: RP_ID,
    });
    if (!verification.verified || !verification.registrationInfo) {
      return send(res, 200, { verified: false });
    }
    const { credential } = verification.registrationInfo;
    credentials.push({
      id: credential.id,
      publicKey: b64encode(credential.publicKey),
      counter: credential.counter,
      transports: credential.transports ?? [],
    });
    await saveCredentials();
    return send(res, 200, { verified: true });
  }

  if (action === "auth-options" && req.method === "POST") {
    const { allowAny } = JSON.parse((await readBody(req)).toString("utf8") || "{}");
    // allowAny => discoverable-credential flow: no allowCredentials pin, so
    // ANY passkey the authenticator holds for this RP ID can be used. This is
    // the fallback when the pinned credential id is stale on the device
    // (e.g. passkey deleted / different authenticator) and the assertion
    // would otherwise fail with NotAllowedError.
    const options = await generateAuthenticationOptions({
      rpID: RP_ID,
      userVerification: "preferred",
      allowCredentials: allowAny
        ? []
        : credentials.map((c) => ({ id: c.id, transports: c.transports })),
    });
    setChallenge("auth", options.challenge);
    return send(res, 200, options);
  }

  if (action === "auth-verify" && req.method === "POST") {
    refreshCredentials();
    const { response } = JSON.parse((await readBody(req)).toString("utf8") || "{}");
    const cred = credentials.find((c) => c.id === response.id);
    if (!cred) {
      // The authenticator used a passkey this proxy doesn't know (stale or
      // re-registered store). Signal the frontend to re-register.
      return send(res, 200, { verified: false, error: "unknown credential", reregister: true });
    }
    const verification = await verifyAuthenticationResponse({
      response,
      expectedChallenge: takeChallenge("auth"),
      expectedOrigin: ORIGINS,
      expectedRPID: RP_ID,
      credential: {
        id: cred.id,
        publicKey: b64decode(cred.publicKey),
        counter: cred.counter,
        transports: cred.transports,
      },
    });
    if (verification.verified) {
      cred.counter = verification.authenticationInfo.newCounter;
      await saveCredentials();
    }
    return send(res, 200, { verified: verification.verified });
  }

  send(res, 404, { error: "unknown passkey endpoint" });
}
