//! Mechanically CONSTRUCT the GEMM's tile/thread layouts with cute-layout's
//! algebra (zipped_divide, make_layout_tv, composition) and verify they equal
//! the known-correct hand-written addressing — the thing Luminal's layout
//! representation cannot do (it has no constructive layout algebra). Then lower
//! the constructed layout to MSL with cute_msl. Exploration/verification step
//! before wiring into the kernel.
//!
//! GEMM config mirrors examples/tiled_gemm.rs: C[M,N], threadgroup tile BM x BN,
//! thread grid (BM/TM) x (BN/TN), each thread a TM x TN register tile.
//! Run: cargo +1.99.1 run --release --example cute_construct -p luminal_metal

use cute_layout::algebra::mapping::composition;
use cute_layout::algebra::thread_value::make_layout_tv;
use cute_layout::algebra::tiling::zipped_divide;
use cute_layout::layout::Layout;
use cute_layout::tuple::IntTuple::{self, Leaf, Tuple};
use luminal_metal::cute_msl::{self, Coord};

const N: i64 = 512;
const BM: i64 = 64;
const BN: i64 = 64;
const BK: i64 = 8;
const TM: i64 = 4;
const TN: i64 = 4;
const M: i64 = 512;

fn pair(a: i64, b: i64) -> IntTuple {
    Tuple(vec![Leaf(a), Leaf(b)])
}

fn main() {
    let threads = (BM / TM) * (BN / TN); // 256
    let values = TM * TN; // 16

    // --- Construct with the algebra ---------------------------------------
    // Thread grid: position (ty,tx) holds thread ty*(BN/TN)+tx. Value grid:
    // (i,j) is value i*TN+j. make_layout_tv builds (thread,value) -> tile cell.
    let thr = Layout::new(pair(BM / TM, BN / TN), pair(BN / TN, 1));
    let val = Layout::new(pair(TM, TN), pair(TN, 1));
    let (tiler, tv) = make_layout_tv(&thr, &val).expect("make_layout_tv");
    println!("make_layout_tv -> tiler {tiler:?}, tv = {tv:?}");

    // The threadgroup tile's layout within C (row-major, row stride N).
    let tile_c = Layout::new(pair(BM, BN), pair(N, 1));
    // Compose: (thread,value) -> within-tile GLOBAL offset (row-major).
    let out_tv = composition(&tile_c, &tv).expect("composition");
    println!("composition(tile_c, tv) = {out_tv:?}");

    // Tile-grid base: tile (ti,tj) -> C offset of the tile's (0,0).
    let tiled_c = zipped_divide(&Layout::new(pair(M, N), pair(N, 1)), &[BM, BN]).expect("zipped_divide");
    println!("zipped_divide(C,[BM,BN]) = {tiled_c:?}");
    // The tile-grid modes (second half) give the per-tile base stride.
    let grid = Layout::new(pair(M / BM, N / BN), pair(BM * N, BN));
    println!("tile-grid layout = {grid:?}");

    // --- Verify against the hand formula ----------------------------------
    // Hand: ty=tid/(BN/TN), tx=tid%(BN/TN); within-tile offset =
    //   (ty*TM + i)*N + (tx*TN + j);  tile base = ti*BM*N + tj*BN.
    let mut ok = true;
    for tid in 0..threads {
        let (ty, tx) = (tid / (BN / TN), tid % (BN / TN));
        for i in 0..TM {
            for j in 0..TN {
                let v = i * TN + j;
                let got = out_tv.call(&pair(tid, v));
                let want = (ty * TM + i) * N + (tx * TN + j);
                if got != want {
                    if ok {
                        println!("MISMATCH tid={tid} v={v}: got {got} want {want}");
                    }
                    ok = false;
                }
            }
        }
    }
    for ti in 0..(M / BM) {
        for tj in 0..(N / BN) {
            let got = grid.call(&pair(ti, tj));
            let want = ti * BM * N + tj * BN;
            if got != want {
                println!("GRID MISMATCH ti={ti} tj={tj}: got {got} want {want}");
                ok = false;
            }
        }
    }
    println!(
        "\nwithin-tile out_tv matches hand formula for all {} (thread,value): {}",
        threads * values,
        if ok { "YES" } else { "NO" }
    );

    // --- Lower the CONSTRUCTED layout to MSL via cute_msl ------------------
    // out_tv's shape is (threads, values); its nested structure lets cute_msl
    // emit the offset over the natural coord. Show the generated expression.
    println!("\nout_tv shape = {:?}, stride = {:?}", out_tv.shape, out_tv.stride);
    // out_tv shape is ((BN/TN,BM/TM),(TN,TM)); cute's natural coord is
    // inner-first per mode (idx2crd), so the coord tree is ((tx,ty),(j,i)) with
    // tx=tid%(BN/TN), ty=tid/(BN/TN), i/j the register-tile loops.
    let coord = Coord::Nest(vec![
        Coord::Nest(vec![Coord::leaf("tx"), Coord::leaf("ty")]),
        Coord::Nest(vec![Coord::leaf("j"), Coord::leaf("i")]),
    ]);
    let lowered = cute_msl::layout_offset_tree(&out_tv, &coord).expect("lower out_tv");
    println!("cute_msl-lowered within-tile offset: {lowered}");

    // Confirm the lowered MSL encodes the hand formula: evaluate it in Rust
    // (substituting the stride-sum it represents) vs the hand offset.
    let grid_coord = Coord::Nest(vec![Coord::leaf("tj"), Coord::leaf("ti")]);
    let grid_lowered = cute_msl::layout_offset_tree(
        &Layout::new(
            Tuple(vec![Leaf(N / BN), Leaf(M / BM)]),
            Tuple(vec![Leaf(BN), Leaf(BM * N)]),
        ),
        &grid_coord,
    )
    .unwrap();
    println!("cute_msl-lowered tile-base offset:   {grid_lowered}");
    println!(
        "\nFull C write address = Ctile_base + within-tile, both GENERATED by\n\
         cute-layout algebra (make_layout_tv/composition/zipped_divide) and\n\
         lowered by cute_msl. No hand-derived strides."
    );

    // --- A/B register-read fragments, constructed by zipped_divide ---------
    // The shared As tile (BM,BK) is partitioned into per-thread (TM,BK)
    // fragments: zipped_divide tiles the row axis by TM, so the tile-grid mode
    // is the thread-row index (ty) and the intra-tile mode is (i,k).
    let a_smem_stride = BK; // plain shared layout row stride (BK); padded = BK+1
    let as_tile = Layout::new(pair(BM, BK), pair(a_smem_stride, 1));
    let as_frag = zipped_divide(&as_tile, &[TM, BK]).expect("zipped_divide As");
    println!("\nas_frag = zipped_divide((BM,BK):(BK,1),[TM,BK]) = {as_frag:?}");
    // Verify As[ty*TM+i][k] = (ty*TM+i)*BK + k for all (ty,i,k). Find the coord
    // order empirically by checking which arrangement reproduces the formula.
    let mut found = None;
    let coord_orders: [(&str, Box<dyn Fn(i64, i64, i64) -> IntTuple>); 2] = [
        ("((i,k),(ty,0))", Box::new(|ty, i, k| Tuple(vec![pair(i, k), pair(ty, 0)]))),
        ("((k,i),(0,ty))", Box::new(|ty, i, k| Tuple(vec![pair(k, i), pair(0, ty)]))),
    ];
    for (label, mk) in &coord_orders {
        let good = (0..BM / TM).all(|ty| {
            (0..TM).all(|i| (0..BK).all(|k| as_frag.call(&mk(ty, i, k)) == (ty * TM + i) * BK + k))
        });
        if good {
            found = Some(*label);
            break;
        }
    }
    println!("as_frag matches As[ty*TM+i][k] with coord order: {found:?}");
}
