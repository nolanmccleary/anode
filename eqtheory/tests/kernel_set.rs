use std::fs;
use std::path::Path;

fn token(parts: &[&str], sep: &str) -> String {
    parts.join(sep)
}

fn scan(dir: &Path, needles: &[String], allow: &str, bad: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) != Some("cute-layout") {
                scan(&path, needles, allow, bad);
            }
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        if path.file_name().and_then(|n| n.to_str()) == Some(allow) {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap();
        for n in needles {
            if text.contains(n.as_str()) {
                bad.push(format!("{} contains `{}`", path.display(), n));
            }
        }
    }
}

#[test]
fn kernels_live_in_one_place_only() {
    let needles = vec![
        token(&["new", "library", "with", "source"], "_"),
        token(&["new", "compute", "pipeline", "state", "with", "function"], "_"),
        token(&["kernel", "void"], " "),
    ];
    let mut bad = Vec::new();
    scan(Path::new("src"), &needles, "metal_backend.rs", &mut bad);
    assert!(
        bad.is_empty(),
        "kernel definitions must live only in src/metal_backend.rs so both arms \
         share one kernel set (RULES.md rule 1): {bad:?}"
    );
}
