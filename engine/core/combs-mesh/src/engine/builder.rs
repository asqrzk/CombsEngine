//! [`Emoji`] — an ordered set of typed blocks — and the fluent
//! [`EmojiBuilder`] matching the spec quick start:
//!
//! ```no_run
//! use combs_mesh::EmojiBuilder;
//! let emoji = EmojiBuilder::new("my-emoji")
//!     .description("...")
//!     .add_todo("task1", "Build the thing")
//!     .add_image_rgba(64, 64, vec![0u8; 64 * 64 * 4])
//!     .with_agent_lifecycle()
//!     .build();
//! ```

use crate::blocks::{
    Block, EncryptionBlock, ImageBlock, LifecycleBlock, LifecycleState, LifecycleTransition,
    SpriteAtlas, TextBlock, TodoBlock, TodoItem, TodoStatus,
};

/// An emoji: a name plus an ordered list of typed blocks.
///
/// The binary/unicode containers carry only the blocks; `name` round-trips
/// through the text block (every builder-made emoji has one).
#[derive(Debug, Clone, PartialEq)]
pub struct Emoji {
    /// Emoji name (mirrored into the text block by [`EmojiBuilder`]).
    pub name: String,
    /// The blocks, in insertion order.
    pub blocks: Vec<Block>,
}

impl Emoji {
    /// The first text block, if any.
    #[must_use]
    pub fn get_text(&self) -> Option<&TextBlock> {
        self.blocks.iter().find_map(|b| match b {
            Block::Txt(t) => Some(t),
            _ => None,
        })
    }

    /// The first image block, if any.
    #[must_use]
    pub fn get_image(&self) -> Option<&ImageBlock> {
        self.blocks.iter().find_map(|b| match b {
            Block::Img(i) => Some(i),
            _ => None,
        })
    }

    /// The first encryption directive, if any.
    #[must_use]
    pub fn get_encryption(&self) -> Option<&EncryptionBlock> {
        self.blocks.iter().find_map(Block::as_encryption)
    }

    /// Iterates over all blocks.
    pub fn iter(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter()
    }

    /// Rebuilds an emoji from decoded blocks, deriving the name from the
    /// text block (empty when there is none).
    pub(crate) fn from_blocks(blocks: Vec<Block>) -> Emoji {
        let name = blocks
            .iter()
            .find_map(|b| match b {
                Block::Txt(t) => Some(t.name.clone()),
                _ => None,
            })
            .unwrap_or_default();
        Emoji { name, blocks }
    }
}

/// Fluent builder. `build()` is infallible: validation happens in the
/// `add_*` methods, which repair inconsistent input (documented per
/// method) instead of panicking.
#[derive(Debug, Clone)]
pub struct EmojiBuilder {
    emoji: Emoji,
}

impl EmojiBuilder {
    /// Starts a builder; every emoji carries a text block so the name
    /// round-trips through the binary/unicode containers.
    #[must_use]
    pub fn new(name: &str) -> Self {
        EmojiBuilder {
            emoji: Emoji {
                name: name.to_string(),
                blocks: vec![Block::Txt(TextBlock {
                    name: name.to_string(),
                    description: String::new(),
                    specs: Vec::new(),
                })],
            },
        }
    }

    /// Sets the description on the text block.
    #[must_use]
    pub fn description(mut self, description: &str) -> Self {
        if let Some(Block::Txt(t)) = self
            .emoji
            .blocks
            .iter_mut()
            .find(|b| matches!(b, Block::Txt(_)))
        {
            t.description = description.to_string();
        }
        self
    }

    /// Adds a todo item (status `Pending`), appending to an existing todo
    /// block when present.
    #[must_use]
    pub fn add_todo(mut self, key: &str, value: &str) -> Self {
        let item = TodoItem {
            key: key.to_string(),
            value: value.to_string(),
            status: TodoStatus::Pending,
            depends_on: Vec::new(),
        };
        if let Some(Block::Tdo(t)) = self
            .emoji
            .blocks
            .iter_mut()
            .find(|b| matches!(b, Block::Tdo(_)))
        {
            t.items.push(item);
        } else {
            self.emoji.blocks.push(Block::Tdo(TodoBlock {
                items: vec![item],
            }));
        }
        self
    }

    /// Adds a single-frame image block from raw RGBA8 pixels. If `rgba`
    /// does not match `width * height * 4` it is zero-padded/truncated to
    /// fit (the builder never panics; strict validation lives in
    /// [`SpriteAtlas::validate`]).
    #[must_use]
    pub fn add_image_rgba(mut self, width: u32, height: u32, mut rgba: Vec<u8>) -> Self {
        rgba.resize(width as usize * height as usize * 4, 0);
        self.emoji.blocks.push(Block::Img(ImageBlock {
            name: String::new(),
            atlas: SpriteAtlas {
                width,
                height,
                frame_width: width.max(1),
                frame_height: height.max(1),
                frame_count: 1,
                rgba,
            },
        }));
        self
    }

    /// Adds the default agent lifecycle: `idle` (initial), `active`,
    /// `sleeping`, with `wake`/`sleep`/`activate`/`deactivate` transitions.
    #[must_use]
    pub fn with_agent_lifecycle(mut self) -> Self {
        self.emoji.blocks.push(Block::Lfc(LifecycleBlock {
            states: vec![
                LifecycleState {
                    name: "idle".into(),
                    initial: true,
                },
                LifecycleState {
                    name: "active".into(),
                    initial: false,
                },
                LifecycleState {
                    name: "sleeping".into(),
                    initial: false,
                },
            ],
            transitions: vec![
                LifecycleTransition {
                    from: "idle".into(),
                    to: "active".into(),
                    event: "activate".into(),
                },
                LifecycleTransition {
                    from: "active".into(),
                    to: "idle".into(),
                    event: "deactivate".into(),
                },
                LifecycleTransition {
                    from: "active".into(),
                    to: "sleeping".into(),
                    event: "sleep".into(),
                },
                LifecycleTransition {
                    from: "sleeping".into(),
                    to: "idle".into(),
                    event: "wake".into(),
                },
            ],
        }));
        self
    }

    /// Adds an encryption directive (which algorithm, which block types to
    /// encrypt at rest).
    #[must_use]
    pub fn encryption(mut self, encryption: EncryptionBlock) -> Self {
        self.emoji.blocks.push(Block::Enc(encryption));
        self
    }

    /// Appends any block.
    #[must_use]
    pub fn add_block(mut self, block: Block) -> Self {
        self.emoji.blocks.push(block);
        self
    }

    /// Finishes building. Infallible by design (see type docs).
    #[must_use]
    pub fn build(self) -> Emoji {
        self.emoji
    }
}
