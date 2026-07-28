/** Type definitions for combs-zerotrust. */

export interface Identity {
  signPub: JsonWebKey;
  signPriv: JsonWebKey;
  encPub: JsonWebKey;
  encPriv: JsonWebKey;
  fingerprint: string;
}

export interface Envelope {
  v: 1;
  from: string;
  eph: JsonWebKey;
  nonce: string;
  ct: string;
  sig: string;
}

export interface ManifestEntry {
  nonce: string;
  wrappedKey: string;
  wrapNonce: string;
  sha256: string;
}

export interface TokenRecord {
  token: string;
  agentId: string;
  expires: number;
}

export declare function b64encode(bytes: Uint8Array): string;
export declare function b64decode(s: string): Uint8Array;
export declare function randomBytes(n: number): Uint8Array;

export declare function generateIdentity(): Promise<Identity>;
export declare function fingerprintOf(signPubJwk: JsonWebKey, encPubJwk: JsonWebKey): Promise<string>;

export declare function seal(opts: {
  from: string;
  fromSignPriv: JsonWebKey;
  toEncPub: JsonWebKey;
  plaintext: string;
}): Promise<Envelope>;

export declare function open(opts: {
  toEncPriv: JsonWebKey;
  fromSignPub: JsonWebKey;
  envelope: Envelope;
}): Promise<string>;

export declare function generateMasterKey(): Uint8Array;
export declare function encryptBlob(
  masterKey: Uint8Array,
  data: Uint8Array,
): Promise<{ blob: string; entry: ManifestEntry }>;
export declare function decryptBlob(
  masterKey: Uint8Array,
  blob: string,
  entry: ManifestEntry,
): Promise<Uint8Array>;

export declare function mintToken(agentId: string, ttlMs?: number): TokenRecord;
export declare function tokenValid(record: TokenRecord | undefined, agentId: string): boolean;
export declare function tokenEncrypt(
  token: string,
  data: Uint8Array,
): Promise<{ nonce: string; ct: string }>;
export declare function tokenDecrypt(
  token: string,
  sealed: { nonce: string; ct: string },
): Promise<Uint8Array>;
