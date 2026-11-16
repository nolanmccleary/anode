//! CuTe-layout -> MSL lowering.
//!
//! The beginning of "CuTe as the kernel IR" for Metal (there is no CuTe->Metal
//! codegen upstream — NVIDIA's CuTe lowering targets PTX/NVVM). Given a
//! `cute_layout` `Layout` (shape:stride) and optional `Swizzle`, emit the Metal
//! Shading Language expression that maps a coordinate to a memory offset, so
//! kernels GENERATE their addressing from a CuTe layout object instead of
//! hand-writing index arithmetic.
//!
//! What it lowers:
//!   * a flat `Layout` -> `crd2idx` as `sum_i coord_i * stride_i` (CuTe's
//!     offset function, Cecka 2.4);
//!   * a `Swizzle{B,M,S}` -> the XOR bit-permutation `x ^ ((x & y_mask) >> S)`
//!     (Cecka 2.4.2 F2 strides) — the layout Luminal's IntExpr cannot express.
//!
//! Scope: flat (all-leaf) layouts, which covers dense tiles and the GEMM/
//! shared-memory layouts here. Hierarchical (nested) modes are a later
//! extension (CuTe's `crd2idx` over nested coordinates).

use anyhow::{Result, bail};
use cute_layout::layout::Layout;
use cute_layout::swizzle::Swizzle;
use cute_layout::tuple::IntTuple;

/// One MSL term `coord * stride`, with CuTe's residues elided: a stride-0
/// (broadcast) axis contributes nothing; a stride-1 axis is the bare coordinate.
fn term(coord: &str, stride: i64) -> Option<String> {
    match stride {
        0 => None,
        1 => Some(coord.to_string()),
        _ => Some(format!("{coord} * {stride}")),
    }
}

fn join(terms: Vec<String>) -> String {
    if terms.is_empty() {
        "0".to_string()
    } else {
        format!("({})", terms.join(" + "))
    }
}

/// MSL expression for `crd2idx(coords)` of a FLAT CuTe layout: the affine map
/// `sum_i coords[i] * stride_i`. `coords` are MSL integer variable names, one
/// per flattened shape mode. (For a hierarchical layout, use
/// [`layout_offset_tree`], or pass one coord per leaf in flatten order.)
pub fn layout_offset(layout: &Layout, coords: &[&str]) -> String {
    let strides = layout.stride.flatten();
    assert_eq!(
        strides.len(),
        coords.len(),
        "cute_msl: {} coords for a rank-{} layout",
        coords.len(),
        strides.len()
    );
    join(
        coords
            .iter()
            .zip(&strides)
            .filter_map(|(c, &s)| term(c, s))
            .collect(),
    )
}

/// A coordinate as a tree of MSL integer expressions, mirroring a CuTe shape's
/// nesting — a *natural coordinate*. `Leaf("i")` is a FREE axis (a loop
/// variable that appears in the offset); `Fixed("ty")` is a SLICED axis (a
/// loop-invariant expression folded into a base offset by [`slice`]); `Nest`
/// groups sub-axes to match a nested shape mode like `((2,3),4)`.
#[derive(Debug, Clone)]
pub enum Coord {
    Leaf(String),
    Fixed(String),
    Nest(Vec<Coord>),
}

impl Coord {
    pub fn leaf(name: &str) -> Coord {
        Coord::Leaf(name.to_string())
    }
    pub fn fixed(expr: &str) -> Coord {
        Coord::Fixed(expr.to_string())
    }
    pub fn nest(children: Vec<Coord>) -> Coord {
        Coord::Nest(children)
    }
}

/// Partial evaluation (CuTe slicing, Cecka §2.5.1). Splits a layout's offset
/// into `(base, free)`: `base` is the contribution of the FIXED (sliced)
/// coordinates — loop-invariant, computed ONCE (e.g. a per-thread base
/// pointer) — and `free` is the offset over the remaining loop coordinates.
/// `base + free` equals the full `crd2idx`. This recovers hand-written inner-
/// loop efficiency: instead of recomputing a thread offset every access, the
/// thread slices its sub-layout and the loop indexes only the free modes.
pub fn slice(layout: &Layout, coord: &Coord) -> Result<(String, String)> {
    let mut base = Vec::new();
    let mut free = Vec::new();
    collect_sliced(&layout.stride, coord, &mut base, &mut free)?;
    Ok((join(base), join(free)))
}

fn collect_sliced(
    stride: &IntTuple,
    coord: &Coord,
    base: &mut Vec<String>,
    free: &mut Vec<String>,
) -> Result<()> {
    match (stride, coord) {
        (IntTuple::Leaf(s), Coord::Leaf(name)) => {
            if let Some(t) = term(name, *s) {
                free.push(t);
            }
            Ok(())
        }
        (IntTuple::Leaf(s), Coord::Fixed(expr)) => {
            if let Some(t) = term(expr, *s) {
                base.push(t);
            }
            Ok(())
        }
        (IntTuple::Tuple(ss), Coord::Nest(cs)) => {
            if ss.len() != cs.len() {
                bail!(
                    "cute_msl::slice: coordinate nesting (rank {}) does not match the stride mode (rank {})",
                    cs.len(),
                    ss.len()
                );
            }
            for (s, c) in ss.iter().zip(cs) {
                collect_sliced(s, c, base, free)?;
            }
            Ok(())
        }
        _ => bail!("cute_msl::slice: coordinate is not congruent to the stride tree"),
    }
}

/// MSL expression for `crd2idx` of a HIERARCHICAL CuTe layout. CuTe's `crd2idx`
/// is multilinear: the offset is `sum over leaves of (leaf coord * leaf
/// stride)`, recursing in lockstep over the stride tree and the coordinate
/// tree (which must be congruent — same nesting). This is the nested
/// generalization of [`layout_offset`]; a tiled layout like
/// `((8, M/8), (8, N/8)) : (...)` addressed by a `(tile, intra-tile)` natural
/// coordinate lowers through here.
pub fn layout_offset_tree(layout: &Layout, coord: &Coord) -> Result<String> {
    let mut terms = Vec::new();
    collect_terms(&layout.stride, coord, &mut terms)?;
    Ok(join(terms))
}

fn collect_terms(stride: &IntTuple, coord: &Coord, out: &mut Vec<String>) -> Result<()> {
    match (stride, coord) {
        (IntTuple::Leaf(s), Coord::Leaf(name) | Coord::Fixed(name)) => {
            if let Some(t) = term(name, *s) {
                out.push(t);
            }
            Ok(())
        }
        (IntTuple::Tuple(ss), Coord::Nest(cs)) => {
            if ss.len() != cs.len() {
                bail!(
                    "cute_msl: coordinate nesting (rank {}) does not match the layout's stride mode (rank {})",
                    cs.len(),
                    ss.len()
                );
            }
            for (s, c) in ss.iter().zip(cs) {
                collect_terms(s, c, out)?;
            }
            Ok(())
        }
        _ => bail!("cute_msl: coordinate is not congruent to the layout's stride tree"),
    }
}

/// MSL expression for a `Swizzle` applied to a (pure, integer) base-offset
/// expression: `base ^ ((base & y_mask) >> S)` (or `<<` for negative shift),
/// matching `Swizzle::apply`. `base` is repeated (it is a pure arithmetic
/// expression; the MSL compiler CSEs it). Emits `u`-suffixed literals for the
/// mask/shift so the bitwise ops stay in `uint`.
pub fn swizzle_offset(sw: &Swizzle, base: &str) -> String {
    if sw.num_bits == 0 {
        return base.to_string();
    }
    let y_mask = sw.y_mask();
    if sw.num_shift >= 0 {
        format!(
            "({base} ^ (({base} & {y_mask}u) >> {s}u))",
            s = sw.num_shift
        )
    } else {
        format!(
            "({base} ^ (({base} & {y_mask}u) << {s}u))",
            s = -sw.num_shift
        )
    }
}

/// Offset of a layout whose physical placement is a swizzle of its affine
/// offset: `swizzle(crd2idx(coords))`.
pub fn swizzled_layout_offset(layout: &Layout, sw: &Swizzle, coords: &[&str]) -> String {
    swizzle_offset(sw, &layout_offset(layout, coords))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cute_layout::tuple::IntTuple::{Leaf, Tuple};

    fn pair(a: i64, b: i64) -> cute_layout::tuple::IntTuple {
        Tuple(vec![Leaf(a), Leaf(b)])
    }

    #[test]
    fn affine_offsets() {
        // row-major [32,32]: stride [32,1] -> row*32 + col.
        let l = Layout::new(pair(32, 32), pair(32, 1));
        assert_eq!(layout_offset(&l, &["row", "col"]), "(row * 32 + col)");
        // padded [32,32]: stride [33,1] -> row*33 + col.
        let p = Layout::new(pair(32, 32), pair(33, 1));
        assert_eq!(layout_offset(&p, &["row", "col"]), "(row * 33 + col)");
    }

    #[test]
    fn swizzle_matches_apply_numerically() {
        // The generated MSL, evaluated in Rust, must equal Swizzle::apply for
        // every (row,col): proves the lowering is faithful.
        let l = Layout::new(pair(32, 32), pair(32, 1));
        let sw = Swizzle::new(5, 0, 5);
        let y_mask = sw.y_mask();
        for row in 0..32i64 {
            for col in 0..32i64 {
                let base = row * 32 + col;
                // mirror swizzle_offset's formula (num_shift >= 0):
                let generated = base ^ ((base & y_mask) >> sw.num_shift);
                assert_eq!(generated, sw.apply(base), "row {row} col {col}");
            }
        }
        // And the emitted string is the expected XOR form.
        let base = layout_offset(&l, &["row", "col"]);
        assert_eq!(
            swizzle_offset(&sw, &base),
            "((row * 32 + col) ^ (((row * 32 + col) & 992u) >> 5u))"
        );
    }

    #[test]
    fn nested_crd2idx_matches_cute() {
        // Hierarchical layout ((2,3),4):((1,8),2). crd2idx(((a,b),c)) = a+8b+2c.
        let shape = Tuple(vec![Tuple(vec![Leaf(2), Leaf(3)]), Leaf(4)]);
        let stride = Tuple(vec![Tuple(vec![Leaf(1), Leaf(8)]), Leaf(2)]);
        let layout = Layout::new(shape, stride);
        let coord = Coord::Nest(vec![
            Coord::Nest(vec![Coord::leaf("a"), Coord::leaf("b")]),
            Coord::leaf("c"),
        ]);
        assert_eq!(
            layout_offset_tree(&layout, &coord).unwrap(),
            "(a + b * 8 + c * 2)"
        );
        // Semantics: CuTe's crd2idx (.call) equals the sum the MSL encodes.
        for a in 0..2i64 {
            for b in 0..3 {
                for c in 0..4 {
                    let numeric = Tuple(vec![Tuple(vec![Leaf(a), Leaf(b)]), Leaf(c)]);
                    assert_eq!(layout.call(&numeric), a + 8 * b + 2 * c, "({a},{b},{c})");
                }
            }
        }
    }

    #[test]
    fn tiled_layout_address_matches_flat_rowmajor() {
        // A 32x32 row-major matrix as 8x8 tiles: natural coord
        // ((ti,mi),(tj,nj)) with ti/tj the tile index, mi/nj intra-tile. The
        // nested layout ((4,8),(4,8)):((256,32),(8,1)) gives crd2idx =
        // ti*256 + mi*32 + tj*8 + nj = the flat row-major address.
        let shape = Tuple(vec![
            Tuple(vec![Leaf(4), Leaf(8)]),
            Tuple(vec![Leaf(4), Leaf(8)]),
        ]);
        let stride = Tuple(vec![
            Tuple(vec![Leaf(256), Leaf(32)]),
            Tuple(vec![Leaf(8), Leaf(1)]),
        ]);
        let layout = Layout::new(shape, stride);
        let coord = Coord::Nest(vec![
            Coord::Nest(vec![Coord::leaf("ti"), Coord::leaf("mi")]),
            Coord::Nest(vec![Coord::leaf("tj"), Coord::leaf("nj")]),
        ]);
        assert_eq!(
            layout_offset_tree(&layout, &coord).unwrap(),
            "(ti * 256 + mi * 32 + tj * 8 + nj)"
        );
        for ti in 0..4i64 {
            for mi in 0..8 {
                for tj in 0..4 {
                    for nj in 0..8 {
                        let numeric = Tuple(vec![
                            Tuple(vec![Leaf(ti), Leaf(mi)]),
                            Tuple(vec![Leaf(tj), Leaf(nj)]),
                        ]);
                        let flat = (ti * 8 + mi) * 32 + (tj * 8 + nj);
                        assert_eq!(layout.call(&numeric), flat, "tile ({ti},{mi},{tj},{nj})");
                    }
                }
            }
        }
    }

    #[test]
    fn congruence_is_checked() {
        let layout = Layout::new(pair(2, 3), pair(3, 1));
        // Leaf coord where the stride is a Tuple -> error.
        assert!(layout_offset_tree(&layout, &Coord::leaf("x")).is_err());
        // Wrong rank (1 vs 2) -> error.
        assert!(layout_offset_tree(&layout, &Coord::Nest(vec![Coord::leaf("a")])).is_err());
    }

    #[test]
    fn slice_partitions_base_and_free() {
        // The A-fragment layout ((TM,BK),(BM/TM,1)):((8,1),(32,0)). Slicing the
        // thread (ty) out: base = ty*32 (computed once), free = i*8 + k (loop).
        let layout = Layout::new(
            Tuple(vec![pair(4, 8), pair(16, 1)]),
            Tuple(vec![pair(8, 1), pair(32, 0)]),
        );
        let coord = Coord::Nest(vec![
            Coord::Nest(vec![Coord::leaf("i"), Coord::leaf("k")]),
            Coord::Nest(vec![Coord::fixed("ty"), Coord::fixed("0")]),
        ]);
        let (base, free) = slice(&layout, &coord).unwrap();
        assert_eq!(base, "(ty * 32)");
        assert_eq!(free, "(i * 8 + k)");
        // Full offset == base + free for all (ty,i,k).
        let full = Coord::Nest(vec![
            Coord::Nest(vec![Coord::leaf("i"), Coord::leaf("k")]),
            Coord::Nest(vec![Coord::leaf("ty"), Coord::leaf("0")]),
        ]);
        assert_eq!(layout_offset_tree(&layout, &full).unwrap(), "(i * 8 + k + ty * 32)");
    }
}
