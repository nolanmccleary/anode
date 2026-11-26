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
            if name != "target" {
                rs_files(&path, out);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    prost_build::compile_protos(&["proto/onnx.proto"], &["proto"]).expect("compile onnx.proto");

    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let mlx_root = format!("{manifest}/.venv/lib/python3.11/site-packages/mlx");
    let mlx_inc = format!("{mlx_root}/include");
    let mlx_lib = format!("{mlx_root}/lib");
    println!("cargo:rerun-if-changed=mlxshim/llama.cpp");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("mlxshim/llama.cpp")
        .include(&mlx_inc)
        .compile("mlxshim");
    println!("cargo:rustc-link-search=native={mlx_lib}");
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:mlxlib={mlx_lib}");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=bin");
    println!("cargo:rerun-if-changed=tests");
    let mut files = Vec::new();
    rs_files(Path::new("src"), &mut files);
    rs_files(Path::new("bin"), &mut files);
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
    let loader_allow = ["native_model.rs", "onnx_import.rs"];

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

    let banned = ["cute", "wire"].concat();
    let mut banned_hits: Vec<String> = Vec::new();
    let mut stack = vec![PathBuf::from(".")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                if name != "target" && name != ".git" {
                    stack.push(path);
                }
                continue;
            }
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !["rs", "toml", "md", "py", "txt", "lock"].contains(&ext) {
                continue;
            }
            if let Ok(text) = fs::read_to_string(&path) {
                if text.contains(&banned) || name.contains(&banned) {
                    banned_hits.push(path.display().to_string());
                }
            }
        }
    }
    if !banned_hits.is_empty() {
        panic!(
            "  [no-{banned}] the token `{banned}` is banned repo-wide and must never reappear — found in:\n    {}",
            banned_hits.join("\n    ")
        );
    }

    for dir in ["src", "bin"] {
        for f in fs::read_dir(dir).unwrap() {
            let path = f.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let text = fs::read_to_string(&path).unwrap_or_default();
            if text.to_lowercase().contains("luminal") {
                panic!(
                    "  [verticalized] {} mentions `luminal` — src/ and bin/ are the system, derived entirely from the CuTe axioms; comparison code that consumes Luminal lives only in harness/luminal/",
                    path.display()
                );
            }
        }
    }

    let fused_allowlist = ["softmax_rows", "rmsnorm_row", "attn_decode", "attn_online", "attn_flash"];
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

    // === canonical-cute-egraph (RULES.md) ===
    // The ONLY things admissible on the e-graph are (1) semimodule-tagged LAYOUT
    // terms (a layout is a function domain->codomain whose stride lives in an
    // integer-semimodule; the semimodule is a first-class equivalence attribute),
    // and (2) the layout EQUIVALENCE RELATIONS Cris Cecka DEFINES in the
    // definitive paper (arXiv:2603.02298): composition, complement, coalesce,
    // logical/zipped division, (left/right) inverse, products. If a method is not
    // a defined functional equivalence relation in that paper, it must NOT be an
    // e-graph operation. Every datatype/relation/function/sort an egglog program
    // declares must be one of these; invented op-datatypes (the former `Val` with
    // Contract/Concat/Mul/Add/Sum/Max/Pt/... and ad-hoc distributivity/softmax
    // rewrites) are BANNED. See [[cute-layout-is-a-function]], Cecka definitive.
    let universal_egraph_decls = [
        "LT",       // layout term datatype (CLit/Compose) — Cecka 3.3 composition
        "Data",     // a layout under refinement
        "Spec",     // a tiling spec — zipped_divide (Cecka 3.5.2), a defined relation
        "Serviced", // result of cute-tile-mode (zipped_divide)
    ];
    // egglog primitives (add_primitive!) must be Cecka-defined layout operations.
    let canonical_cute_primitives = [
        "cute-compose", "cute-coalesce", "cute-complement",
        "cute-divide", "cute-tile-mode", "cute-product",
        "cute-inverse", "cute-left-inverse", "cute-right-inverse",
    ];
    let egraph_decls = ["(relation ", "(datatype ", "(function ", "(sort "];

    let mut bad: Vec<String> = Vec::new();
    for f in &files {
        let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let text = fs::read_to_string(f).unwrap();
        let at = |rule: &str| format!("  [{rule}] {}", f.display());

        for decl in egraph_decls {
            for (i, _) in text.match_indices(decl) {
                let rest = &text[i + decl.len()..];
                let term: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                    .collect();
                if !term.is_empty() && !universal_egraph_decls.contains(&term.as_str()) {
                    bad.push(format!(
                        "{} (declares e-graph term `{term}` — the e-graph admits ONLY semimodule-tagged layout terms + Cecka-defined equivalence relations; invented op-datatypes are banned)",
                        at("canonical-cute-egraph")
                    ));
                }
            }
        }
        for (i, _) in text.match_indices("add_primitive!") {
            let rest = &text[i..];
            if let Some(q0) = rest.find('"') {
                let after = &rest[q0 + 1..];
                if let Some(q1) = after.find('"') {
                    let prim = &after[..q1];
                    if !canonical_cute_primitives.contains(&prim) {
                        bad.push(format!(
                            "{} (e-graph primitive `{prim}` — only Cecka-defined CuTe layout operations may back e-graph rewrites)",
                            at("canonical-cute-egraph")
                        ));
                    }
                }
            }
        }

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
