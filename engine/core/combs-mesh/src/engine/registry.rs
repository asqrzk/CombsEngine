//! Content-addressed emoji store.
//!
//! Layout (mirrors the model cache convention — `$COMBS_HOME` respected,
//! else `~/.cache/combs`):
//!
//! ```text
//! $COMBS_HOME/mesh/            (default ~/.cache/combs/mesh)
//! ├── <sha256-hex>.cmse        one file per registered emoji binary
//! └── index.json               { name: { hash, bytes, block_count } }
//! ```
//!
//! The hash is SHA-256 of the *plaintext* `.cmse` binary (same digest
//! family as the zerotrust manifest hashing). `index.json` is a cache of
//! convenience: a missing/corrupt index is rebuilt by scanning the
//! directory.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::engine::{Emoji, EmojiExporter};
use crate::error::{MeshError, Result};

/// One entry in the registry.
#[derive(Debug, Clone, PartialEq)]
pub struct RegistryEntry {
    /// Registered name (from the emoji's text block).
    pub name: String,
    /// SHA-256 hex of the binary.
    pub hash: String,
    /// Path of the `.cmse` file.
    pub path: PathBuf,
    /// Size of the binary in bytes.
    pub bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    hash: String,
    bytes: usize,
    block_count: usize,
}

/// The registry. Cheap to construct; all state lives on disk.
#[derive(Debug, Clone)]
pub struct Registry {
    root: PathBuf,
}

impl Registry {
    /// Opens the default registry (`$COMBS_HOME/mesh`, else
    /// `~/.cache/combs/mesh`), creating the directory if needed.
    pub fn open() -> Result<Registry> {
        Registry::open_at(mesh_root()?)
    }

    /// Opens a registry rooted at an explicit directory (tests, custom
    /// deployments).
    pub fn open_at(root: PathBuf) -> Result<Registry> {
        fs::create_dir_all(&root)?;
        Ok(Registry { root })
    }

    /// The registry root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Registers `emoji`: writes `<sha256>.cmse` and records the name in
    /// the index. Returns the hash. Idempotent.
    pub fn register(&self, emoji: &Emoji) -> Result<String> {
        let binary = EmojiExporter::to_binary(emoji)?;
        let hash = sha256_hex(&binary);
        fs::write(self.root.join(format!("{hash}.cmse")), &binary)?;
        let mut index = self.load_index();
        let name = if emoji.name.is_empty() {
            hash[..12].to_string()
        } else {
            emoji.name.clone()
        };
        index.insert(
            name,
            IndexEntry {
                hash: hash.clone(),
                bytes: binary.len(),
                block_count: emoji.blocks.len(),
            },
        );
        self.save_index(&index)?;
        Ok(hash)
    }

    /// Resolves a name or a 64-char hex hash to an emoji.
    pub fn resolve(&self, name_or_hash: &str) -> Result<Emoji> {
        let index = self.load_index();
        let hash = match index.get(name_or_hash) {
            Some(entry) => entry.hash.clone(),
            None if is_sha256_hex(name_or_hash) => name_or_hash.to_string(),
            None => {
                return Err(MeshError::Registry(format!(
                    "no emoji named '{name_or_hash}'"
                )));
            }
        };
        let path = self.root.join(format!("{hash}.cmse"));
        let bytes = fs::read(&path)
            .map_err(|e| MeshError::Registry(format!("cannot read {}: {e}", path.display())))?;
        Ok(EmojiExporter::from_binary(&bytes)?)
    }

    /// Lists all registered emojis. Rebuilds the index from the directory
    /// when it is missing or corrupt.
    pub fn list(&self) -> Result<Vec<RegistryEntry>> {
        let index = self.load_index();
        let mut entries: Vec<RegistryEntry> = index
            .iter()
            .map(|(name, e)| RegistryEntry {
                name: name.clone(),
                hash: e.hash.clone(),
                path: self.root.join(format!("{}.cmse", e.hash)),
                bytes: e.bytes,
            })
            .collect();
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    /// Removes a name from the index; deletes the `.cmse` file when no
    /// other name references the same hash. Returns whether the name
    /// existed.
    pub fn remove(&self, name: &str) -> Result<bool> {
        let mut index = self.load_index();
        let Some(entry) = index.remove(name) else {
            return Ok(false);
        };
        if !index.values().any(|e| e.hash == entry.hash) {
            let _ = fs::remove_file(self.root.join(format!("{}.cmse", entry.hash)));
        }
        self.save_index(&index)?;
        Ok(true)
    }

    /// Loads the index; rebuilds from the directory on missing/corrupt.
    fn load_index(&self) -> HashMap<String, IndexEntry> {
        let path = self.root.join("index.json");
        if let Ok(bytes) = fs::read(&path) {
            if let Ok(index) = serde_json::from_slice::<HashMap<String, IndexEntry>>(&bytes) {
                return index;
            }
        }
        self.rebuild_index()
    }

    fn rebuild_index(&self) -> HashMap<String, IndexEntry> {
        let mut index = HashMap::new();
        if let Ok(dir) = fs::read_dir(&self.root) {
            for file in dir.flatten() {
                let path = file.path();
                if path.extension().and_then(|e| e.to_str()) != Some("cmse") {
                    continue;
                }
                let Ok(bytes) = fs::read(&path) else { continue };
                let Ok(emoji) = EmojiExporter::from_binary(&bytes) else {
                    continue;
                };
                let hash = sha256_hex(&bytes);
                let name = if emoji.name.is_empty() {
                    hash[..12].to_string()
                } else {
                    emoji.name
                };
                index.insert(
                    name,
                    IndexEntry {
                        hash,
                        bytes: bytes.len(),
                        block_count: emoji.blocks.len(),
                    },
                );
            }
        }
        let _ = self.save_index(&index);
        index
    }

    fn save_index(&self, index: &HashMap<String, IndexEntry>) -> Result<()> {
        let json = serde_json::to_vec_pretty(index)?;
        fs::write(self.root.join("index.json"), json)?;
        Ok(())
    }
}

/// The default registry root: `$COMBS_HOME/mesh`, else
/// `$HOME/.cache/combs/mesh` (mirrors `combs pull`'s cache resolution).
pub fn mesh_root() -> Result<PathBuf> {
    let root = std::env::var("COMBS_HOME")
        .map(PathBuf::from)
        .or_else(|_| {
            std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .map(|h| PathBuf::from(h).join(".cache/combs"))
        })
        .map_err(|_| MeshError::Registry("cannot locate a home directory (set COMBS_HOME)".into()))?;
    Ok(root.join("mesh"))
}

/// SHA-256 hex digest (content address).
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{b:02x}"));
    }
    hex
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
