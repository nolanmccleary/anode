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
    let loader_allow = ["native_model.rs"];

    let allowed_pointwise = ["Exp2", "Recip", "Sqrt", "Rsqrt", "Relu", "Step", "CastIdentity"];
    let ksrc = fs::read_to_string("src/kernels.rs").unwrap_or_default();
    let enum_body = |name: &str| -> Vec<String> {
        let tag = format!("pub enum {name} {{");
        let start = ksrc.find(&tag).map(|i| i + tag.len()).unwrap_or_else(|| {
            panic!("  [node-types] src/kernels.rs must declare `pub enum {name}` — the e-graph vocabulary is fixed")
        });
        let end = start + ksrc[start..].find('}').unwrap();
        ksrc[start..end]
            .split(',')
            .map(|v| v.split('(').next().unwrap().trim().to_string())
            .filter(|v| !v.is_empty())
            .collect()
    };
    let node_types = enum_body("NodeType");
    if node_types != ["Move", "Pointwise"] {
        panic!(
            "  [node-types] NodeType must be exactly {{Move, Pointwise}} — got {node_types:?}. The e-graph vocabulary is LAYOUT MOVES (default; non-injective write = contraction over the fixed (+,x) semiring) plus POINTWISE NONLINEARITIES. No other node type may exist; compose, don't invent."
        );
    }
    for v in enum_body("Pointwise") {
        if !allowed_pointwise.contains(&v.as_str()) {
            panic!(
                "  [node-types] Pointwise variant `{v}` is not in the closed nonlinearity list {allowed_pointwise:?}. Extending the list is a conscious act: add it here WITH justification that it is a true pointwise nonlinearity, not a composition."
            );
        }
    }

    let manifest = fs::read_to_string("Cargo.toml").unwrap_or_default();
    if manifest.contains("luminal-ai") {
        panic!(
            "  [verticalized] Cargo.toml references the luminal-ai org — the engine pins must point upstream (egraphs-good/egglog); nothing of ours may depend on their repos."
        );
    }

    for f in fs::read_dir("src").unwrap() {
        let path = f.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap_or_default();
        if text.to_lowercase().contains("luminal") {
            panic!(
                "  [verticalized] {} mentions `luminal` — src/ is the system, derived entirely from the CuTe axioms; comparison code that consumes Luminal lives only in baseline/",
                path.display()
            );
        }
    }

    let fused_allowlist = ["softmax_rows", "rmsnorm_row", "attn_decode", "attn_online"];
    let kernels_src = fs::read_to_string("src/kernels.rs").unwrap_or_default();
    let kdef = ["kernel", "void "].join(" ");
    for block in kernels_src.split(&kdef).skip(1) {
        let name: String = block.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
        let body_end = block.find(&kdef).unwrap_or(block.len());
        let body = &block[..body_end];
        let transcendental = ["exp(", "exp2(", "sqrt("].iter().any(|t| body.contains(t));
        let reduction = body.contains("for ") && body.contains("+=");
        if transcendental && reduction && !fused_allowlist.contains(&name.as_str()) {
            panic!(
                "  [no-hamfisted-kernels] src/kernels.rs kernel `{name}` fuses a reduction with a nonlinearity but is not a registered optimization member. Fused kernels must (1) be added to the allowlist in build.rs AND (2) have a working primitive-only path proven by tests/general_path.rs — never be the sole implementation of an op."
            );
        }
    }

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
