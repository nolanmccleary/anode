use std::process::Command;

#[test]
fn luminal_core_is_unmodified() {
    let out = Command::new("git")
        .args(["status", "--porcelain", "--", "refs/luminal"])
        .current_dir("..")
        .output();

    let Ok(out) = out else {
        eprintln!("git unavailable; skipping luminal-pristine check");
        return;
    };
    if !out.status.success() {
        eprintln!("not a git repo; skipping luminal-pristine check");
        return;
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let mut touched = Vec::new();
    for line in text.lines() {
        if line.len() < 4 {
            continue;
        }
        let status = &line[..2];
        let path = line[3..].trim();
        if status == "??" {
            continue;
        }
        if path.contains("/target/") {
            continue;
        }
        if path.contains("refs/luminal/") {
            touched.push(format!("{status} {path}"));
        }
    }

    assert!(
        touched.is_empty(),
        "RULES.md rule 2: Luminal core must stay pristine, but these tracked files \
         under refs/luminal are modified: {touched:?}"
    );
}
