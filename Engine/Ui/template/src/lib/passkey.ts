/**
 * Passkey ceremonies (SimpleWebAuthn browser client).
 *
 * The passkey approves permission requests: the dialog runs an
 * authentication ceremony and the proxy only records the grant after
 * verifying the assertion server-side. When WebAuthn is unavailable the
 * app degrades to click-approval (documented; the crypto integrity layer
 * is unaffected either way).
 */

import {
  browserSupportsWebAuthn,
  startAuthentication,
  startRegistration,
} from "@simplewebauthn/browser";

export const webauthnSupported = browserSupportsWebAuthn();

export async function passkeyStatus(): Promise<boolean> {
  try {
    const res = await fetch("/api/auth/passkey/status");
    return res.ok && (await res.json()).registered === true;
  } catch {
    return false;
  }
}

/** Registers a new device passkey (called once from the setup flow). */
export async function registerPasskey(): Promise<boolean> {
  try {
    const optRes = await fetch("/api/auth/passkey/register-options", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ username: "combs-user" }),
    });
    if (!optRes.ok) return false;
    const response = await startRegistration({ optionsJSON: await optRes.json() });
    const verRes = await fetch("/api/auth/passkey/register-verify", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ response }),
    });
    const ok = verRes.ok && (await verRes.json()).verified === true;
    if (ok) sessionStorage.setItem("combs.passkey", "1"); // never ask again this session
    return ok;
  } catch {
    return false;
  }
}

/** Proves user presence for a permission approval. */
export async function approveWithPasskey(): Promise<boolean> {
  try {
    // First attempt: pin to the credential id the proxy has on record.
    let response = await authenticate(false);
    if (!response) {
      // NotAllowedError: the authenticator doesn't hold that credential id
      // (stale / re-registered / different authenticator). Retry WITHOUT the
      // allowCredentials pin so any passkey for this RP ID can be used.
      response = await authenticate(true);
    }
    if (!response) return false;
    const verRes = await fetch("/api/auth/passkey/auth-verify", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ response }),
    });
    return verRes.ok && (await verRes.json()).verified === true;
  } catch {
    return false;
  }
}

/** Runs one authentication ceremony; returns null when the user/agent
 *  cannot or will not produce an assertion (NotAllowedError etc.). */
async function authenticate(allowAny: boolean) {
  try {
    const optRes = await fetch("/api/auth/passkey/auth-options", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ allowAny }),
    });
    if (!optRes.ok) return null;
    return await startAuthentication({ optionsJSON: await optRes.json() });
  } catch {
    return null;
  }
}
