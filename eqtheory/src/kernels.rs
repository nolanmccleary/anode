pub const TILE: u64 = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EqClass {
    Matmul,
    Copy,
    Gather,
    Relu,
    Add,
    Softmax,
}

pub const KERNELS: &[(&str, EqClass)] = &[
    ("tiled_matmul", EqClass::Matmul),
    ("materialize", EqClass::Copy),
    ("gather3", EqClass::Gather),
    ("relu", EqClass::Relu),
    ("add2", EqClass::Add),
    ("softmax_rows", EqClass::Softmax),
];

pub const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 16;

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
