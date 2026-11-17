use eqtheory::kernels::TILE;
use eqtheory::metal_backend::CuteMetal;
use metal::MTLSize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn inline_includes(path: &Path, seen: &mut HashSet<PathBuf>) -> String {
    let canon = path.canonicalize().expect("source exists");
    if !seen.insert(canon.clone()) {
        return String::new();
    }
    let dir = canon.parent().unwrap().to_path_buf();
    let mut out = String::new();
    for line in std::fs::read_to_string(&canon).unwrap().lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("#include \"") {
            let name = rest.split('"').next().unwrap();
            let mut cand = dir.join(name);
            if !cand.exists() {
                let roots = [
                    dir.join("../../include"),
                    dir.join("../include"),
                    dir.join("../../include/common"),
                    dir.join("../../include/types"),
                    dir.join("../../include/ops"),
                ];
                match roots.iter().map(|r| r.join(name)).find(|p| p.exists()) {
                    Some(c) => cand = c,
                    None => {
                        eprintln!("skipping missing include {name} (from {dir:?})");
                        continue;
                    }
                }
            }
            out.push_str(&inline_includes(&cand, seen));
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn main() {
    let backend = CuteMetal::new();
    let tm_root = Path::new("../refs/ThunderMittens/ThunderMittens");
    let src = inline_includes(
        &tm_root.join("kernels/matmul_custom/matmul_custom.metal"),
        &mut HashSet::new(),
    );
    let tm_pipe = backend.external_pipeline(&src, true, "matmul_custom_float32");
    let ours_fast = backend.external_pipeline(eqtheory::kernels::KERNEL_SOURCE, true, "tiled_matmul");

    println!("TM matmul_custom_float32 compiled (fast-math ON, their default).");
    println!("ours = kernels.rs tiled_matmul (fast-math OFF, as used in the A/B).\n");
    println!(
        "{:<22} {:>10} {:>10} {:>11} {:>9} {:>9} {:>8}",
        "shape N,K,M", "TM GF/s", "ours GF/s", "ours+fm GF/s", "TM ms", "ours ms", "match"
    );
    println!("{}", "-".repeat(74));

    for (n, k, m) in [
        (1024usize, 1024usize, 1024usize),
        (1024, 2752, 1024),
        (1024, 1024, 2752),
        (2048, 2048, 2048),
        (32, 1024, 32000),
    ] {
        assert!(n % 32 == 0 && m % 32 == 0 && k % 16 == 0, "TM divisibility");
        let a_host: Vec<f32> = (0..n * k).map(|i| ((i * 37 + 11) % 97) as f32 / 97.0 - 0.5).collect();
        let b_host: Vec<f32> = (0..k * m).map(|i| ((i * 53 + 7) % 89) as f32 / 89.0 - 0.5).collect();
        let a = backend.upload(&a_host);
        let b = backend.upload(&b_host);
        let d_tm = backend.alloc_bytes(n * m * 4);
        let d_ours = backend.alloc_bytes(n * m * 4);

        let run_tm = || {
            let cb = backend.owned_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&tm_pipe);
            enc.set_buffer(0, Some(&d_tm), 0);
            enc.set_buffer(1, Some(&a), 0);
            enc.set_buffer(2, Some(&b), 0);
            let (ni, ki, mi) = (n as i32, k as i32, m as i32);
            enc.set_bytes(3, 4, &ni as *const i32 as *const _);
            enc.set_bytes(4, 4, &ki as *const i32 as *const _);
            enc.set_bytes(5, 4, &mi as *const i32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((m / 32) as u64, (n / 32) as u64, 1),
                MTLSize::new(32, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        let run_ours = || {
            let cb = backend.owned_command_buffer();
            backend.matmul_into(
                &cb,
                &a,
                &b,
                &d_ours,
                n,
                k,
                m,
                [0, k as i32, 1, 0],
                [0, m as i32, 1, 0],
                [0, m as i32, 1, 0],
            );
            cb.commit();
            cb.wait_until_completed();
        };

        run_tm();
        run_ours();
        let tm_vals = backend.read(&d_tm, n * m);
        let our_vals = backend.read(&d_ours, n * m);
        let peak = our_vals.iter().fold(1e-30f32, |x, &v| x.max(v.abs()));
        let maxd = tm_vals
            .iter()
            .zip(&our_vals)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let ok = maxd / peak < 1e-3;

        let mut tm_ms = f64::INFINITY;
        for _ in 0..20 {
            let t = Instant::now();
            run_tm();
            tm_ms = tm_ms.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let mut our_ms = f64::INFINITY;
        for _ in 0..20 {
            let t = Instant::now();
            run_ours();
            our_ms = our_ms.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let run_fast = || {
            let cb = backend.owned_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ours_fast);
            enc.set_buffer(0, Some(&a), 0);
            enc.set_buffer(1, Some(&b), 0);
            enc.set_buffer(2, Some(&d_ours), 0);
            let (mu, ku, nu) = (n as u32, k as u32, m as u32);
            enc.set_bytes(3, 4, &mu as *const u32 as *const _);
            enc.set_bytes(4, 4, &ku as *const u32 as *const _);
            enc.set_bytes(5, 4, &nu as *const u32 as *const _);
            let ap = [0i32, k as i32, 1, 0];
            let bp = [0i32, m as i32, 1, 0];
            let cp = [0i32, m as i32, 1, 0];
            enc.set_bytes(6, 16, ap.as_ptr() as *const _);
            enc.set_bytes(7, 16, bp.as_ptr() as *const _);
            enc.set_bytes(8, 16, cp.as_ptr() as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new((m as u64).div_ceil(16), (n as u64).div_ceil(16), 1),
                MTLSize::new(16, 16, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        };
        run_fast();
        let mut fast_ms = f64::INFINITY;
        for _ in 0..20 {
            let t = Instant::now();
            run_fast();
            fast_ms = fast_ms.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let flops = 2.0 * n as f64 * k as f64 * m as f64;
        println!(
            "{:<22} {:>10.1} {:>10.1} {:>11.1} {:>9.3} {:>9.3} {:>8}",
            format!("{n},{k},{m}"),
            flops / (tm_ms / 1e3) / 1e9,
            flops / (our_ms / 1e3) / 1e9,
            flops / (fast_ms / 1e3) / 1e9,
            tm_ms,
            our_ms,
            if ok { "ok" } else { "DIFF" }
        );
        let _ = TILE;
    }
}
