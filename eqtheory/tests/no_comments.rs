use std::fs;
use std::path::Path;

fn marker(a: char, b: char) -> String {
    [a, b].iter().collect()
}

fn scan(dir: &Path, line: &str, block: &str, bad: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) != Some("cute-layout") {
                scan(&path, line, block, bad);
            }
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let text = fs::read_to_string(&path).unwrap();
            if text.contains(line) || text.contains(block) {
                bad.push(path.display().to_string());
            }
        }
    }
}

#[test]
fn project_code_has_no_comments() {
    let line = marker('/', '/');
    let block = marker('/', '*');
    let mut bad = Vec::new();
    scan(Path::new("src"), &line, &block, &mut bad);
    scan(Path::new("tests"), &line, &block, &mut bad);
    scan(Path::new("core-port"), &line, &block, &mut bad);
    assert!(bad.is_empty(), "comment markers found in project code: {bad:?}");
}
