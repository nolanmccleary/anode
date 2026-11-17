use eqtheory::arena::{allocate_arena, read_f32, upload};
use eqtheory::metal_backend::CuteMetal;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, GraphTensor, NodeIndex};
use luminal::shape::IntExpr;
use luminal_metal::{CompileOptions, HostBuffer, MetalRuntime};
use metal::Buffer;
use std::time::Instant;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * 0.01
        })
        .collect()
}

fn mix(be: &CuteMetal, x_in: &Buffer, n: usize, steps: usize, affine: bool) -> Vec<f32> {
    objc::rc::autoreleasepool(|| {
        let ni = n as i32;
        let cb = be.command_buffer();
        let mut keep: Vec<Buffer> = Vec::new();
        let mut x = x_in.clone();
        for _ in 0..steps {
            let y = be.alloc(n * n);
            keep.push(y.clone());
            if affine {
                let xt = be.alloc(n * n);
                keep.push(xt.clone());
                be.materialize_into(cb, &x, &xt, n, n, [0, 1, ni, 0]);
                be.add2_into(cb, &x, &xt, &y, n, n, [0, ni, 1, 0]);
            } else {
                be.add2_into(cb, &x, &x, &y, n, n, [0, 1, ni, 0]);
            }
            be.relu_into(cb, &y, n * n);
            x = y;
        }
        cb.commit();
        cb.wait_until_completed();
        be.read(&x, n * n)
    })
}

struct Oracle {
    rt: MetalRuntime,
    arena: eqtheory::arena::Allocation,
    out: NodeIndex,
}

fn build_luminal(n: usize, steps: usize, input: &[f32]) -> Oracle {
    let ni = n as i64;
    let mut g = Graph::new();
    let x = g.tensor([IntExpr::from(ni), IntExpr::from(ni)], DType::F32);
    let mut cur: GraphTensor = x;
    for _ in 0..steps {
        cur = (cur + cur.transpose(0, 1)).relu();
    }
    let mut rt = MetalRuntime::load(&g).unwrap();
    let search_inputs: FxHashMap<NodeIndex, HostBuffer> =
        [(x.id, input.to_vec().into())].into_iter().collect();
    let opts = CompileOptions {
        generations: 3,
        generation_size: 4,
        trials: 2,
        seed: 0,
        search_log: false,
        ..CompileOptions::default()
    };
    rt.search(&Default::default(), &Default::default(), &search_inputs, &opts)
        .unwrap();
    let mut arena = allocate_arena(&rt).unwrap();
    upload(&rt, &mut arena, x.id, input.to_vec()).unwrap();
    rt.execute(arena.buffer()).unwrap();
    Oracle { rt, arena, out: cur.id }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn time_min(iters: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e3
}

fn main() {
    let n = 2048usize;
    println!("MEMORY-BOUND A/B: cute vs affine on ONE shared kernel set");
    println!("model = relu(x + x^T) stacked M times, N={n} (all elementwise, memory-bound)");
    println!("cute reads x^T by stride; affine materializes it. same add2 + relu kernels.\n");
    println!(
        "{:>3} | {:>6} {:>6} {:>7} | {:>9} {:>9} | {:>9} {:>9} {:>7} | {:>6}",
        "M", "cuteK", "affK", "copies", "cute ms", "aff ms", "c-vs-a", "c-vs-lum", "speedup", "oracle"
    );
    println!("{}", "-".repeat(96));

    let be = CuteMetal::new();
    let input = fill(n * n, 1);
    let x_buf = be.upload(&input);
    let mut all_ok = true;
    for m in [1usize, 2, 4, 8] {
        let cute = mix(&be, &x_buf, n, m, false);
        let affine = mix(&be, &x_buf, n, m, true);
        let oracle = build_luminal(n, m, &input);
        let lum = read_f32(&oracle.rt, &oracle.arena, oracle.out).unwrap();

        let peak = lum.iter().fold(1e-30f32, |a, &x| a.max(x.abs()));
        let c_vs_a = max_abs(&cute, &affine);
        let c_vs_lum = max_abs(&cute, &lum) / peak;
        let ok = c_vs_a < 1e-4 && c_vs_lum < 1e-4;
        all_ok &= ok;

        let cute_ms = time_min(15, || {
            std::hint::black_box(mix(&be, &x_buf, n, m, false));
        });
        let aff_ms = time_min(15, || {
            std::hint::black_box(mix(&be, &x_buf, n, m, true));
        });

        println!(
            "{:>3} | {:>6} {:>6} {:>7} | {:>9.3} {:>9.3} | {:>9.2e} {:>9.2e} {:>6.2}x | {:>6}",
            m,
            2 * m,
            3 * m,
            m,
            cute_ms,
            aff_ms,
            c_vs_a,
            c_vs_lum,
            aff_ms / cute_ms,
            if ok { "match" } else { "FAIL" }
        );
    }
    println!("\ncopies = x^T materializations cute elides (1 per step); add2 reads the transpose by stride");
    println!("{}", if all_ok { "ALL MATCH" } else { "MISMATCH" });
    std::process::exit(if all_ok { 0 } else { 1 });
}
