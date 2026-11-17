use anyhow::{bail, Result};
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::dtype::PlanDtype;
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};
use luminal_metal::kernels::{layout_read_index, CodegenCtx, Coords, KernelOp, KernelSource};
use luminal_metal::symbolic::Expr;
use luminal_metal::{MetalOpInterface, RegisteredOp};

pub const TILE: u64 = 16;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum EqClass {
    Matmul,
    Copy,
    Gather,
    Relu,
    Add,
    Softmax,
    MulEw,
    Recip,
    Sqrt,
    Exp2,
    Iota,
    CastI2F,
    Round,
    Scatter,
    ReduceSum,
    ReduceMax,
    LessThan,
    Fill,
    CastB2F,
    Rsqrt,
}

pub const KERNELS: &[(&str, EqClass)] = &[
    ("tiled_matmul", EqClass::Matmul),
    ("materialize", EqClass::Copy),
    ("gather3", EqClass::Gather),
    ("relu", EqClass::Relu),
    ("add2", EqClass::Add),
    ("softmax_rows", EqClass::Softmax),
    ("mul2", EqClass::MulEw),
    ("recip1", EqClass::Recip),
    ("sqrt1", EqClass::Sqrt),
    ("exp2_1", EqClass::Exp2),
    ("iota_i", EqClass::Iota),
    ("cast_i2f", EqClass::CastI2F),
    ("round_f", EqClass::Round),
    ("scatter_rows", EqClass::Scatter),
    ("reduce_sum_rows", EqClass::ReduceSum),
    ("reduce_max_rows", EqClass::ReduceMax),
    ("less_than_i", EqClass::LessThan),
    ("fill1", EqClass::Fill),
    ("copy_strided", EqClass::Copy),
    ("copy_strided_i", EqClass::Copy),
    ("add_strided", EqClass::Add),
    ("mul_strided", EqClass::MulEw),
    ("recip_o", EqClass::Recip),
    ("sqrt_o", EqClass::Sqrt),
    ("exp2_o", EqClass::Exp2),
    ("cast_b2f", EqClass::CastB2F),
    ("gather4", EqClass::Gather),
    ("scatter4", EqClass::Scatter),
    ("reduce_sum_axis", EqClass::ReduceSum),
    ("reduce_max_axis", EqClass::ReduceMax),
    ("sqrt_s", EqClass::Sqrt),
    ("recip_s", EqClass::Recip),
    ("exp2_s", EqClass::Exp2),
    ("rsqrt_s", EqClass::Rsqrt),
];

pub const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 16;

static inline long strided_off(constant int* m, uint i) {
    long off = m[1];
    long r = i;
    int nm = m[0];
    for (int k = 0; k < nm; ++k) {
        long sz = m[2 + k];
        off += (r % sz) * (long)m[10 + k];
        r /= sz;
    }
    return off;
}

kernel void copy_strided(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    constant int*       m   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = SRC[strided_off(m, gid)];
}

kernel void copy_strided_i(
    device const int* SRC [[buffer(0)]],
    device int*       DST [[buffer(1)]],
    constant uint&    n   [[buffer(2)]],
    constant int*     m   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = SRC[strided_off(m, gid)];
}

kernel void add_strided(
    device const float* A  [[buffer(0)]],
    device const float* B  [[buffer(1)]],
    device float*       C  [[buffer(2)]],
    constant uint&      n  [[buffer(3)]],
    constant int*       ma [[buffer(4)]],
    constant int*       mb [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    C[gid] = A[strided_off(ma, gid)] + B[strided_off(mb, gid)];
}

kernel void mul_strided(
    device const float* A  [[buffer(0)]],
    device const float* B  [[buffer(1)]],
    device float*       C  [[buffer(2)]],
    constant uint&      n  [[buffer(3)]],
    constant int*       ma [[buffer(4)]],
    constant int*       mb [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    C[gid] = A[strided_off(ma, gid)] * B[strided_off(mb, gid)];
}

kernel void recip_o(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = 1.0f / SRC[gid];
}

kernel void sqrt_o(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = sqrt(SRC[gid]);
}

kernel void exp2_o(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = exp2(SRC[gid]);
}

kernel void sqrt_s(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    constant int*       m   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = sqrt(SRC[strided_off(m, gid)]);
}

kernel void recip_s(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    constant int*       m   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = 1.0f / SRC[strided_off(m, gid)];
}

kernel void exp2_s(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    constant int*       m   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = exp2(SRC[strided_off(m, gid)]);
}

kernel void rsqrt_s(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    constant int*       m   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = 1.0f / sqrt(SRC[strided_off(m, gid)]);
}

kernel void cast_b2f(
    device const unsigned char* SRC [[buffer(0)]],
    device float*               DST [[buffer(1)]],
    constant uint&              n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = SRC[gid] ? 1.0f : 0.0f;
}

kernel void gather4(
    device const float* DATA [[buffer(0)]],
    device const int*   C0   [[buffer(1)]],
    device const int*   C1   [[buffer(2)]],
    device const int*   C2   [[buffer(3)]],
    device const int*   C3   [[buffer(4)]],
    device float*       OUT  [[buffer(5)]],
    constant int*       meta [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int rank = meta[0];
    uint n = uint(meta[1]);
    if (gid >= n) return;
    long off = meta[6];
    if (rank > 0) off += (long)C0[gid] * meta[2];
    if (rank > 1) off += (long)C1[gid] * meta[3];
    if (rank > 2) off += (long)C2[gid] * meta[4];
    if (rank > 3) off += (long)C3[gid] * meta[5];
    OUT[gid] = DATA[off];
}

kernel void scatter4(
    device const float* SRC  [[buffer(0)]],
    device const int*   C0   [[buffer(1)]],
    device const int*   C1   [[buffer(2)]],
    device const int*   C2   [[buffer(3)]],
    device const int*   C3   [[buffer(4)]],
    device float*       OUT  [[buffer(5)]],
    constant int*       meta [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int rank = meta[0];
    uint n = uint(meta[1]);
    if (gid >= n) return;
    long off = meta[6];
    if (rank > 0) off += (long)C0[gid] * meta[2];
    if (rank > 1) off += (long)C1[gid] * meta[3];
    if (rank > 2) off += (long)C2[gid] * meta[4];
    if (rank > 3) off += (long)C3[gid] * meta[5];
    OUT[off] = SRC[gid];
}

kernel void reduce_sum_axis(
    device const float* X    [[buffer(0)]],
    device float*       OUT  [[buffer(1)]],
    constant uint&      outer [[buffer(2)]],
    constant uint&      axis  [[buffer(3)]],
    constant uint&      inner [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= outer * inner) return;
    uint o = gid / inner;
    uint in_ = gid % inner;
    float s = 0.0;
    for (uint a = 0; a < axis; ++a) s += X[(o*axis + a)*inner + in_];
    OUT[gid] = s;
}

kernel void reduce_max_axis(
    device const float* X    [[buffer(0)]],
    device float*       OUT  [[buffer(1)]],
    constant uint&      outer [[buffer(2)]],
    constant uint&      axis  [[buffer(3)]],
    constant uint&      inner [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= outer * inner) return;
    uint o = gid / inner;
    uint in_ = gid % inner;
    float m = -INFINITY;
    for (uint a = 0; a < axis; ++a) m = max(m, X[(o*axis + a)*inner + in_]);
    OUT[gid] = m;
}

kernel void tiled_matmul(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       C   [[buffer(2)]],
    constant uint&      M   [[buffer(3)]],
    constant uint&      K   [[buffer(4)]],
    constant uint&      N   [[buffer(5)]],
    constant int4&      ap  [[buffer(6)]],
    constant int4&      bp  [[buffer(7)]],
    constant int4&      cp  [[buffer(8)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];
    int ao = ap.x, ars = ap.y, acs = ap.z;
    int bo = bp.x, brs = bp.y, bcs = bp.z;
    int co = cp.x, crs = cp.y, ccs = cp.z;
    uint row = tgp.y*TILE + tid.y;
    uint col = tgp.x*TILE + tid.x;
    float acc = 0.0;
    uint ntiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < ntiles; ++t) {
        uint aK = t*TILE + tid.x;
        uint bK = t*TILE + tid.y;
        As[tid.y][tid.x] = (row < M && aK < K) ? A[ao + int(row)*ars + int(aK)*acs] : 0.0;
        Bs[tid.y][tid.x] = (bK < K && col < N) ? B[bo + int(bK)*brs + int(col)*bcs] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = 0; e < TILE; ++e) acc += As[tid.y][e] * Bs[e][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[co + int(row)*crs + int(col)*ccs] = acc;
}

kernel void gather3(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    constant uint4&     ext [[buffer(2)]],
    constant int4&      st  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint e0 = ext.x, e1 = ext.y, e2 = ext.z;
    uint total = e0 * e1 * e2;
    if (gid >= total) return;
    uint c = gid % e2;
    uint t = gid / e2;
    uint b = t % e1;
    uint a = t / e1;
    DST[gid] = SRC[st.x + int(a)*st.y + int(b)*st.z + int(c)*st.w];
}

kernel void materialize(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      cols [[buffer(3)]],
    constant int4&      sp   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.y;
    uint j = gid.x;
    if (i >= rows || j >= cols) return;
    int so = sp.x, srs = sp.y, scs = sp.z;
    DST[i*cols + j] = SRC[so + int(i)*srs + int(j)*scs];
}

kernel void relu(
    device float*  X [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    X[gid] = max(0.0f, X[gid]);
}

kernel void add2(
    device const float* A    [[buffer(0)]],
    device const float* B    [[buffer(1)]],
    device float*       C    [[buffer(2)]],
    constant uint&      rows [[buffer(3)]],
    constant uint&      cols [[buffer(4)]],
    constant int4&      bp   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.y;
    uint j = gid.x;
    if (i >= rows || j >= cols) return;
    C[i*cols + j] = A[i*cols + j] + B[bp.x + int(i)*bp.y + int(j)*bp.z];
}

kernel void mul2(
    device const float* A    [[buffer(0)]],
    device const float* B    [[buffer(1)]],
    device float*       C    [[buffer(2)]],
    constant uint&      rows [[buffer(3)]],
    constant uint&      cols [[buffer(4)]],
    constant int4&      bp   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.y;
    uint j = gid.x;
    if (i >= rows || j >= cols) return;
    C[i*cols + j] = A[i*cols + j] * B[bp.x + int(i)*bp.y + int(j)*bp.z];
}

kernel void recip1(
    device float*  X [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    X[gid] = 1.0f / X[gid];
}

kernel void sqrt1(
    device float*  X [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    X[gid] = sqrt(X[gid]);
}

kernel void exp2_1(
    device float*  X [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    X[gid] = exp2(X[gid]);
}

kernel void iota_i(
    device int*    DST  [[buffer(0)]],
    constant uint& n    [[buffer(1)]],
    constant int&  off  [[buffer(2)]],
    constant int&  step [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = off + int(gid)*step;
}

kernel void cast_i2f(
    device const int* SRC [[buffer(0)]],
    device float*     DST [[buffer(1)]],
    constant uint&    n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = float(SRC[gid]);
}

kernel void round_f(
    device float*  X [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    X[gid] = rint(X[gid]);
}

kernel void scatter_rows(
    device const float* SRC  [[buffer(0)]],
    device float*       DST  [[buffer(1)]],
    device const int*   IDX  [[buffer(2)]],
    constant uint&      rows [[buffer(3)]],
    constant uint&      cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = rows * cols;
    if (gid >= total) return;
    uint r = gid / cols;
    uint c = gid % cols;
    DST[uint(IDX[r])*cols + c] = SRC[gid];
}

kernel void reduce_sum_rows(
    device const float* X    [[buffer(0)]],
    device float*       OUT  [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    float s = 0.0;
    for (uint j = 0; j < cols; ++j) s += X[gid*cols + j];
    OUT[gid] = s;
}

kernel void reduce_max_rows(
    device const float* X    [[buffer(0)]],
    device float*       OUT  [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    float m = -INFINITY;
    for (uint j = 0; j < cols; ++j) m = max(m, X[gid*cols + j]);
    OUT[gid] = m;
}

kernel void less_than_i(
    device const int*     A [[buffer(0)]],
    device const int*     B [[buffer(1)]],
    device unsigned char* C [[buffer(2)]],
    constant uint&        n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    C[gid] = A[gid] < B[gid] ? 1 : 0;
}

kernel void fill1(
    device float*   DST [[buffer(0)]],
    constant uint&  n   [[buffer(1)]],
    constant float& v   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    DST[gid] = v;
}

kernel void softmax_rows(
    device float*  X     [[buffer(0)]],
    constant uint& rows  [[buffer(1)]],
    constant uint& cols  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    device float* row = X + gid*cols;
    float m = -INFINITY;
    for (uint j = 0; j < cols; ++j) m = max(m, row[j]);
    float s = 0.0;
    for (uint j = 0; j < cols; ++j) { float e = exp(row[j] - m); row[j] = e; s += e; }
    for (uint j = 0; j < cols; ++j) row[j] /= s;
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsqrtFunctional;

impl OpSlotNames for RsqrtFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for RsqrtFunctional {
    fn label(&self) -> &str {
        "RsqrtFunctionalGeneric"
    }
}

impl Bufferizable for RsqrtFunctional {}

impl ToDps for RsqrtFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(RsqrtFunctionalDps))
    }
}

impl LayoutIrOp for RsqrtFunctional {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsqrtFunctionalDps;

impl OpSlotNames for RsqrtFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for RsqrtFunctionalDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "RsqrtFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1
    }
}

impl Bufferizable for RsqrtFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for RsqrtFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for RsqrtFunctionalDps {}

impl KernelOp for RsqrtFunctionalDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        if ctx.operand_dtypes[0] != PlanDtype::F32 || ctx.dest_dtypes[0] != PlanDtype::F32 {
            bail!("rsqrt fused kernel is F32 only");
        }
        let out_dims = &ctx.dest_dims[0];
        let n = out_dims
            .iter()
            .cloned()
            .reduce(|a, b| a * b)
            .unwrap_or_else(|| Expr::from(1usize));
        let layout = ctx.operand_layout(0);
        let (code, idx) =
            layout_read_index("a", layout, out_dims, Coords::FlatIndex { prefix: "c" })?;
        if !code.is_empty() || idx != "i" {
            bail!("rsqrt fused kernel requires a contiguous operand");
        }
        let source = format!(
            "kernel void k(device const float* a, device float* out, device const long* params, uint gid [[thread_position_in_grid]]) {{\n    const ulong n = {n};\n    ulong i = gid;\n    if (i < n) out[i] = 1.0f / sqrt(a[i]);\n}}"
        );
        Ok(vec![KernelSource::plain(source, n)])
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RsqrtFunctionalMatcher;

impl OpMatcher for RsqrtFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpRsqrtFunctionalGeneric"
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LayoutOpConstructors,
                text: include_str!("rsqrt_constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("rsqrt_match.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("layout", 1)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(RsqrtFunctional)
    }
}

pub fn shared_registry() -> Vec<RegisteredOp> {
    let mut reg = luminal_metal::metal_registry();
    reg.push(RegisteredOp::new(
        Box::new(RsqrtFunctionalMatcher),
        Box::new(RsqrtFunctional),
    ));
    reg
}
