use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name != "cute-layout" && name != "target" {
                rs_files(&path, out);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=tests");
    let mut files = Vec::new();
    rs_files(Path::new("src"), &mut files);
    rs_files(Path::new("tests"), &mut files);
    for f in &files {
        println!("cargo:rerun-if-changed={}", f.display());
    }

    let line_comment = ['/', '/'].iter().collect::<String>();
    let block_comment = ['/', '*'].iter().collect::<String>();
    let kernel_def = ["kernel", "void"].join(" ");
    let make_lib = ["new", "library", "with", "source"].join("_");
    let make_pipe = ["new", "compute", "pipeline", "state", "with", "function"].join("_");
    let graph_new = ["Graph", "new"].join("::");
    let arm_flags = [
        ["affine", "bool"].join(": "),
        ["arm", "bool"].join(": "),
        ["cute", "bool"].join(": "),
    ];
    let graph_builders = [
        graph_new.clone(),
        [".split", "dims("].join("_"),
        [".merge", "dims("].join("_"),
        ".matmul(".to_string(),
        ".softmax(".to_string(),
    ];

    let kernels_allow = ["kernels.rs", "metal_backend.rs"];
    let loader_allow = ["loaders.rs"];

    let mut bad: Vec<String> = Vec::new();
    for f in &files {
        let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let text = fs::read_to_string(f).unwrap();
        let at = |rule: &str| format!("  [{rule}] {}", f.display());

        if text.contains(&line_comment) || text.contains(&block_comment) {
            bad.push(at("no-comments"));
        }
        if !kernels_allow.contains(&name) {
            for needle in [&kernel_def, &make_lib, &make_pipe] {
                if text.contains(needle.as_str()) {
                    bad.push(format!("{} (`{needle}` only in kernels.rs/metal_backend.rs)", at("one-kernel-set")));
                }
            }
        }
        for needle in &arm_flags {
            if text.contains(needle.as_str()) {
                bad.push(format!("{} (`{needle}` — arms come from the pipeline, not a bool)", at("no-arm-hacks")));
            }
        }
        if !loader_allow.contains(&name) {
            for needle in &graph_builders {
                if text.contains(needle.as_str()) {
                    bad.push(format!("{} (`{needle}` — comparison models must be LOADED, not authored inline)", at("no-hand-graphs")));
                }
            }
        }
    }

    let cute_switch_allow = [
        "refs/luminal/Cargo.toml",
        "refs/luminal/src/frontend/movement.rs",
        "refs/luminal/src/logical/ir.rs",
        "refs/luminal/src/extraction.rs",
        "refs/luminal/crates/luminal_metal/Cargo.toml",
        "refs/luminal/crates/luminal_metal/src/bindings.rs",
        "refs/luminal/crates/luminal_metal/src/saturation.rs",
        "refs/luminal/src/logical/op/index_map_apply/compose.egg",
        "refs/luminal/src/logical/op/index_map_apply/mod.rs",
    ];
    if let Ok(out) = Command::new("git")
        .args(["status", "--porcelain", "--", "refs/luminal"])
        .current_dir("..")
        .output()
    {
        if out.status.success() {
            for l in String::from_utf8_lossy(&out.stdout).lines() {
                if l.len() < 4 {
                    continue;
                }
                let (st, p) = (&l[..2], l[3..].trim());
                let allowed = cute_switch_allow.iter().any(|a| p.ends_with(a));
                if st != "??" && !p.contains("/target/") && p.contains("refs/luminal/") && !allowed {
                    bad.push(format!("  [luminal-pristine] {st} {p} (only the gated cute-switch files may change)"));
                }
            }
        }
    }

    if !bad.is_empty() {
        panic!(
            "\n\n=== LINT FAILURES (RULES.md) — build blocked ===\n{}\n",
            bad.join("\n")
        );
    }
}
