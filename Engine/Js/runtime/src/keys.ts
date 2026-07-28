/**
 * KeyRing — zero-trust identity + peer registry for agent channels.
 *
 * One KeyRing per participant (orchestrator / agent). Peers are exchanged
 * once over the token-authed channel at registration; afterwards every
 * message is a sealed envelope (ECDH → AES-GCM + hash + signature) and
 * needs no further permission checks — tampered or wrongly-signed
 * envelopes are rejected by `openFrom`.
 */

import {
  type Envelope,
  generateIdentity,
  type Identity,
  open,
  seal,
} from "@combs/zerotrust";

export interface PeerKeys {
  signPub: JsonWebKey;
  encPub: JsonWebKey;
  fingerprint: string;
}

export class KeyRing {
  private constructor(readonly identity: Identity) {}
  private peers = new Map<string, PeerKeys>(); // keyed by fingerprint

  static async create(): Promise<KeyRing> {
    return new KeyRing(await generateIdentity());
  }

  /** Public half to hand to peers. */
  publicKeys(): PeerKeys {
    return {
      signPub: this.identity.signPub,
      encPub: this.identity.encPub,
      fingerprint: this.identity.fingerprint,
    };
  }

  addPeer(keys: PeerKeys): void {
    this.peers.set(keys.fingerprint, keys);
  }

  peer(fingerprint: string): PeerKeys | undefined {
    return this.peers.get(fingerprint);
  }

  peerCount(): number {
    return this.peers.size;
  }

  /** Seals a JSON-serializable payload for a peer (by fingerprint). */
  async sealFor(peerFingerprint: string, payload: unknown): Promise<Envelope> {
    const peer = this.peers.get(peerFingerprint);
    if (!peer) throw new Error(`keyring: unknown peer ${peerFingerprint}`);
    return await seal({
      from: this.identity.fingerprint,
      fromSignPriv: this.identity.signPriv,
      toEncPub: peer.encPub,
      plaintext: JSON.stringify(payload),
    });
  }

  /** Opens an envelope; sender is authenticated by its registry key. */
  async openFrom(envelope: Envelope): Promise<unknown> {
    const peer = this.peers.get(envelope.from);
    if (!peer) throw new Error(`keyring: envelope from unknown peer ${envelope.from}`);
    const plaintext = await open({
      toEncPriv: this.identity.encPriv,
      fromSignPub: peer.signPub,
      envelope,
    });
    return JSON.parse(plaintext);
  }
}
