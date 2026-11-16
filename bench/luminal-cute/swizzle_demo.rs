//! Atomic swizzle demo — the ONE place CuTe's layout algebra can express a
//! layout Luminal's `IntExpr` cannot, and it matters.
//!
//! Luminal's offset algebra (IntLit/IntVar/CoordVar/Add/Mul/TruncDiv/TruncRem/
//! CeilDiv/Min/Max) has NO bitwise/XOR operator, so it can express PLAIN and
//! PADDED shared-memory layouts but NOT an XOR SWIZZLE. CuTe (F2 swizzle strides)
//! can. Swizzles matter for threadgroup-memory bank conflicts.
//!
//! This microbenchmark stages a tile in threadgroup memory and reads it in a
//! column pattern that 32-way bank-conflicts under a plain layout. The SAME
//! kernel template is instantiated with three shared-memory index functions —
//! the only difference between runs:
//!   plain   : row*BK + col                      (conflicts; BK = 32 banks)
//!   padded  : row*(BK+1) + col                  (affine's fix; +1 column memory)
//!   swizzle : row*BK + (col XOR (row & (BK-1)))  (CuTe's fix; no extra memory)
//! All three compute identical results (the index is a bijection), so this is
//! purely a layout-representation A/B at the kernel level.
//!
//! Run: cargo +1.99.1 run --release --example swizzle_demo -p luminal_metal
#![cfg(target_os = "macos")]

use luminal_metal::cute_msl;
use metal::{Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const TM: usize = 32; // rows (one per simdgroup lane)
const BK: usize = 32; // columns = bank count
const ITERS: usize = 512; // re-reads of the tile, to amplify shared-mem cost
const GROUPS: usize = 8192; // threadgroups
const INPUT_TILES: usize = 256; // distinct input tiles, indexed by threadgroup

/// The `IDX(row, col)` MSL expression for each layout, GENERATED FROM A CuTe
/// layout object by `cute_msl` (not hand-written), plus the per-row stride
/// (shared-memory footprint = TM * stride floats). `plain`/`padded` are affine
/// layouts (Luminal's IntExpr could also express them); `swizzle` is a CuTe
/// `Swizzle` — the XOR layout Luminal's IntExpr cannot express.
fn cute_index(name: &str) -> (String, usize) {
    use cute_layout::layout::Layout;
    use cute_layout::swizzle::Swizzle;
    use cute_layout::tuple::IntTuple::{Leaf, Tuple};
    let pair = |a: i64, b: i64| Tuple(vec![Leaf(a), Leaf(b)]);
    let (tm, bk) = (TM as i64, BK as i64);
    match name {
        "plain" => (
            cute_msl::layout_offset(&Layout::new(pair(tm, bk), pair(bk, 1)), &["row", "col"]),
            BK,
        ),
        "padded" => (
            cute_msl::layout_offset(&Layout::new(pair(tm, bk), pair(bk + 1, 1)), &["row", "col"]),
            BK + 1,
        ),
        "swizzle" => (
            cute_msl::swizzled_layout_offset(
                &Layout::new(pair(tm, bk), pair(bk, 1)),
                &Swizzle::new(5, 0, 5),
                &["row", "col"],
            ),
            BK,
        ),
        _ => unreachable!(),
    }
}

fn kernel_source(idx: &str, stride: usize) -> String {
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void k(device const float* a, device float* out,
              uint tid [[thread_index_in_threadgroup]],
              uint gid [[threadgroup_position_in_grid]]) {{
    threadgroup float As[{tm} * {stride}];
    uint base = (gid % {ntiles}u) * ({tm}u * {bk}u);
    // Cooperative load: logical (row,col) -> As[IDX] = a[base + row*BK + col].
    for (uint i = tid; i < {tm}u * {bk}u; i += {tm}u) {{
        uint row = i / {bk}u;
        uint col = i % {bk}u;
        As[{idx}] = a[base + row * {bk}u + col];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each lane reads its own row across all columns, ITERS times. The column
    // rotates by `it` so the loads are not hoisted. Under `plain`, the 32 lanes
    // hit one bank per (it,k) -> 32-way conflict.
    float acc = 0.0f;
    uint row = tid;
    for (uint it = 0; it < {iters}u; ++it) {{
        for (uint kk = 0; kk < {bk}u; ++kk) {{
            uint col = (kk + it) & {bkm1}u;
            acc += As[{idx}];
        }}
    }}
    out[gid * {tm}u + tid] = acc;
}}"#,
        tm = TM,
        bk = BK,
        bkm1 = BK - 1,
        stride = stride,
        ntiles = INPUT_TILES,
        iters = ITERS,
        idx = idx,
    )
}

/// Prove the kernel's swizzle IS a CuTe swizzle: `cute_layout::Swizzle` with
/// B=5 (the 5 bits of a 32-wide row), M=0, S=5 (shift the row's low 5 bits down
/// onto the column field) applied to the plain tile offset `row*BK + col`
/// yields exactly `row*BK + (col ^ (row & 31))`. CuTe GENERATES this XOR layout
/// from its algebra; Luminal's IntExpr (no bitwise ops) cannot.
fn verify_cute_generates_the_swizzle() {
    use cute_layout::swizzle::Swizzle;
    let sw = Swizzle::new(5, 0, 5); // B=5, M=0, S=5  (CuTe's S<5,0,5>)
    for row in 0..TM as i64 {
        for col in 0..BK as i64 {
            let plain = row * BK as i64 + col;
            let cute = sw.apply(plain);
            let kernel = row * BK as i64 + (col ^ (row & (BK as i64 - 1)));
            assert_eq!(
                cute, kernel,
                "CuTe swizzle S<5,0,5>.apply({plain}) = {cute} != kernel {kernel}"
            );
        }
    }
    println!(
        "verified: cute_layout::Swizzle::new(5,0,5) generates the kernel's XOR layout\n\
         (row*{BK} + (col ^ (row & {})))  — a layout Luminal's IntExpr cannot express.\n",
        BK - 1
    );
}

fn main() {
    verify_cute_generates_the_swizzle();
    let device = Device::system_default().expect("no Metal device");
    println!("device: {}\n", device.name());
    let queue = device.new_command_queue();

    // Input: INPUT_TILES tiles of TM*BK floats; output: GROUPS*TM floats.
    let a: Vec<f32> = (0..INPUT_TILES * TM * BK)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let a_buf = device.new_buffer_with_data(
        a.as_ptr() as *const _,
        (a.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (GROUPS * TM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let variants = ["plain", "padded", "swizzle"];
    let mut results = Vec::new();
    let mut reference: Option<Vec<f32>> = None;

    println!("CuTe-generated shared-memory index expressions (cute_msl):");
    for name in variants {
        let (idx_expr, _) = cute_index(name);
        println!("  {name:>8}: As[{idx_expr}]");
    }
    println!();

    for name in variants {
        let (idx_expr, stride) = cute_index(name);
        let src = kernel_source(&idx_expr, stride);
        let lib = device
            .new_library_with_source(&src, &metal::CompileOptions::new())
            .unwrap_or_else(|e| panic!("{name} compile: {e}"));
        let func = lib.get_function("k", None).unwrap();
        let pso = device
            .new_compute_pipeline_state_with_function(&func)
            .unwrap();

        // Warmup + timed min over runs (GPU timestamps).
        let mut best = f64::MAX;
        for run in 0..40 {
            let t = Instant::now();
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(GROUPS as u64, 1, 1),
                MTLSize::new(TM as u64, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            if run >= 8 {
                best = best.min(t.elapsed().as_secs_f64());
            }
        }

        // Read back output, verify all variants agree (identical computation).
        let out = unsafe {
            std::slice::from_raw_parts(out_buf.contents() as *const f32, GROUPS * TM).to_vec()
        };
        match &reference {
            None => reference = Some(out),
            Some(r) => {
                let max_err = r
                    .iter()
                    .zip(&out)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(max_err < 1e-2, "{name} disagrees with plain (err {max_err})");
            }
        }
        results.push((name, best * 1e3, stride * TM * 4));
    }

    let plain_ms = results[0].1;
    println!(
        "{:>8} | {:>10} | {:>10} | {:>14} | source",
        "layout", "time (ms)", "vs plain", "tgmem bytes"
    );
    println!("{}", "-".repeat(70));
    for (name, ms, bytes) in &results {
        let expressible = if *name == "swizzle" {
            "CuTe-only (XOR)"
        } else {
            "Luminal IntExpr"
        };
        println!(
            "{:>8} | {:>10.4} | {:>9.2}x | {:>10} B | {}",
            name,
            ms,
            plain_ms / ms,
            bytes,
            expressible
        );
    }
    println!("{}", "-".repeat(70));
    println!(
        "\nplain = 32-way bank conflict (BK=32 banks, column read). padded and\n\
         swizzle both avoid it; padded costs +{} B/threadgroup, swizzle costs 0.\n\
         Luminal's IntExpr can emit plain/padded but NOT the XOR swizzle; CuTe can.\n",
        TM * 4
    );
}
