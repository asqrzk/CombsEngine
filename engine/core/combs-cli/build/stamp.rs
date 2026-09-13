// The source identity a build stamps: git's answer where the source is a
// checkout, else the commit a packed archive says it was made from (a pod
// builds from `git archive`, which carries no .git), else "unknown".
// Included by build.rs and by build_info's tests — one implementation.

struct SourceIdentity {
    commit: String,
    short: String,
    branch: String,
    dirty: bool,
}

fn source_identity(
    git_commit: Option<String>,
    git_short: Option<String>,
    git_branch: Option<String>,
    git_dirty: Option<bool>,
    stamped_commit: Option<String>,
    stamped_ref: Option<String>,
) -> SourceIdentity {
    let unknown = || "unknown".to_string();
    if let Some(commit) = git_commit {
        return SourceIdentity {
            commit,
            short: git_short.unwrap_or_else(unknown),
            branch: git_branch.unwrap_or_else(unknown),
            dirty: git_dirty.unwrap_or(false),
        };
    }
    // Only a commit id is believed: 7 to 40 hex digits, nothing else.
    let stamped = stamped_commit
        .map(|c| c.trim().to_ascii_lowercase())
        .filter(|c| (7..=40).contains(&c.len()) && c.bytes().all(|b| b.is_ascii_hexdigit()));
    match stamped {
        // An archive of a commit is that commit, unmodified by construction.
        Some(commit) => SourceIdentity {
            short: commit[..7].to_string(),
            branch: stamped_ref
                .map(|r| r.trim().to_string())
                .filter(|r| {
                    !r.is_empty()
                        && r.len() <= 100
                        && r.bytes().all(|b| b.is_ascii_alphanumeric() || b"._/-".contains(&b))
                })
                .unwrap_or_else(unknown),
            commit,
            dirty: false,
        },
        None => SourceIdentity {
            commit: unknown(),
            short: unknown(),
            branch: unknown(),
            dirty: false,
        },
    }
}
