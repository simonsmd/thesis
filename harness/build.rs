//! Records what the binaries were built from, for the headers of result files.

use std::{path::Path, process::Command};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");

    // Rebuild when the measured code or the checked out commit changes. Directories are scanned
    // recursively; paths that do not exist would force a rerun on every build.
    for path in ["parser/src", "parser/Cargo.toml", "harness/src", ".git/HEAD", ".git/index", ".git/refs"] {
        if root.join(path).exists() {
            println!("cargo::rerun-if-changed=../{path}");
        }
    }

    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    let rev = git(&["rev-parse", "HEAD"]).unwrap_or_default();
    // new source files count: they change the build as much as edits do
    let dirty = git(&["status", "--porcelain", "--untracked-files=all", "--", "parser", "harness"])
        .is_some_and(|s| !s.is_empty());

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = Command::new(rustc)
        .arg("-V")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default();
    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default().replace('\x1f', " ");

    println!("cargo::rustc-env=HARNESS_GIT_REV={rev}");
    println!("cargo::rustc-env=HARNESS_GIT_DIRTY={dirty}");
    println!("cargo::rustc-env=HARNESS_RUSTC={rustc_version}");
    println!("cargo::rustc-env=HARNESS_RUSTFLAGS={rustflags}");
    println!("cargo::rustc-env=HARNESS_PROFILE={}", std::env::var("PROFILE").unwrap_or_default());
}
