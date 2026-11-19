use eqtheory::metal_backend::CuteMetal;
use std::time::Instant;

fn main() {
    let backend = CuteMetal::new();
    println!("{:<16} {:>10} {:>10}", "shape K,N", "ms", "GB/s");
    for (k, n, rows) in [
        (1024usize, 256usize, false),
        (1024, 1024, false),
        (1024, 2752, false),
        (2752, 1024, false),
        (1024, 32000, true),
    ] {
        let x = backend.upload(&vec![0.5f32; k]);
        let w = backend.upload(&vec![0.25f32; k * n]);
        let out = backend.alloc_bytes(n * 4);
        let (ap, bp, cp) = if rows {
            ([0i32, 0, 1, 0], [0i32, 1, k as i32, 0], [0i32, 1, 1, 0])
        } else {
            ([0i32, 0, 1, 0], [0i32, n as i32, 1, 0], [0i32, n as i32, 1, 0])
        };
        backend.tune_gemv(k, n, &x, &w, ap, bp);
        let run = |iters: usize| {
            let cb = backend.owned_command_buffer();
            for _ in 0..iters {
                backend.matmul_or_gemv_into(&cb, &x, &w, &out, 1, k, n, ap, bp, cp);
            }
            cb.commit();
            cb.wait_until_completed();
        };
        run(5);
        let mut best = f64::INFINITY;
        for _ in 0..10 {
            let t = Instant::now();
            run(10);
            best = best.min(t.elapsed().as_secs_f64() * 1e3 / 10.0);
        }
        println!(
            "{:<16} {:>10.4} {:>10.1}",
            format!("{k},{n}"),
            best,
            (k * n * 4) as f64 / (best / 1e3) / 1e9
        );
    }
}
