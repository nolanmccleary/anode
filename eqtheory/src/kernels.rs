pub const TILE: u64 = 16;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Pointwise {
    Exp2,
    Recip,
    Sqrt,
    Rsqrt,
    Relu,
    Step,
    CastIdentity,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum NodeType {
    Move,
    Pointwise(Pointwise),
}

pub const KERNELS: &[(&str, NodeType)] = &[
    ("tiled_matmul", NodeType::Move),
    ("bmm", NodeType::Move),
    ("materialize", NodeType::Move),
    ("gather3", NodeType::Move),
    ("relu", NodeType::Pointwise(Pointwise::Relu)),
    ("add2", NodeType::Move),
    ("softmax_rows", NodeType::Move),
    ("mul2", NodeType::Move),
    ("recip1", NodeType::Pointwise(Pointwise::Recip)),
    ("sqrt1", NodeType::Pointwise(Pointwise::Sqrt)),
    ("exp2_1", NodeType::Pointwise(Pointwise::Exp2)),
    ("iota_i", NodeType::Move),
    ("cast_i2f", NodeType::Pointwise(Pointwise::CastIdentity)),
    ("round_f", NodeType::Pointwise(Pointwise::CastIdentity)),
    ("scatter_rows", NodeType::Move),
    ("reduce_sum_rows", NodeType::Move),
    ("reduce_max_rows", NodeType::Move),
    ("less_than_i", NodeType::Pointwise(Pointwise::Step)),
    ("fill1", NodeType::Move),
    ("copy_strided", NodeType::Move),
    ("copy_strided_i", NodeType::Move),
    ("add_strided", NodeType::Move),
    ("mul_strided", NodeType::Move),
    ("recip_o", NodeType::Pointwise(Pointwise::Recip)),
    ("sqrt_o", NodeType::Pointwise(Pointwise::Sqrt)),
    ("exp2_o", NodeType::Pointwise(Pointwise::Exp2)),
    ("cast_b2f", NodeType::Pointwise(Pointwise::CastIdentity)),
    ("gather4", NodeType::Move),
    ("scatter4", NodeType::Move),
    ("reduce_sum_axis", NodeType::Move),
    ("reduce_max_axis", NodeType::Move),
    ("sqrt_s", NodeType::Pointwise(Pointwise::Sqrt)),
    ("recip_s", NodeType::Pointwise(Pointwise::Recip)),
    ("exp2_s", NodeType::Pointwise(Pointwise::Exp2)),
    ("rsqrt_s", NodeType::Pointwise(Pointwise::Rsqrt)),
    ("gemv_n4", NodeType::Move),
    ("rmsnorm_row", NodeType::Move),
    ("silu_mul", NodeType::Move),
    ("ew_chain", NodeType::Move),
    ("attn_decode", NodeType::Move),
    ("attn_online", NodeType::Move),
    ("xcopy_f", NodeType::Move),
    ("xcopy_i", NodeType::Move),
    ("xcopy_i2f", NodeType::Pointwise(Pointwise::CastIdentity)),
    ("lt_strided", NodeType::Pointwise(Pointwise::Step)),
    ("reduce_sum_l", NodeType::Move),
    ("reduce_max_l", NodeType::Move),
];

pub const TEMPLATE_KERNELS: &[(&str, NodeType)] = &[
    ("gemm_sg", NodeType::Move),
    ("gemv_t", NodeType::Move),
    ("gemv_rows_t", NodeType::Move),
    ("gemv_sg_t", NodeType::Move),
];

pub const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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

kernel void bmm(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       C   [[buffer(2)]],
    constant uint&      M   [[buffer(3)]],
    constant uint&      K   [[buffer(4)]],
    constant uint&      N   [[buffer(5)]],
    constant int4&      ap  [[buffer(6)]],
    constant int4&      bp  [[buffer(7)]],
    constant int4&      cp  [[buffer(8)]],
    constant int*       bat [[buffer(9)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];
    uint2 tid = uint2(tid3.x, tid3.y);
    long ao = ap.x, bo = bp.x, co = cp.x;
    int nax = bat[0];
    int r = int(tgp.z);
    for (int k = 0; k < nax; ++k) {
        int sz = bat[1 + 4 * k];
        int i = r % sz;
        r /= sz;
        ao += (long)i * bat[2 + 4 * k];
        bo += (long)i * bat[3 + 4 * k];
        co += (long)i * bat[4 + 4 * k];
    }
    int ars = ap.y, acs = ap.z;
    int brs = bp.y, bcs = bp.z;
    int crs = cp.y, ccs = cp.z;
    uint row = tgp.y * TILE + tid.y;
    uint col = tgp.x * TILE + tid.x;
    float acc = 0.0;
    uint ntiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < ntiles; ++t) {
        uint aK = t * TILE + tid.x;
        uint bK = t * TILE + tid.y;
        As[tid.y][tid.x] = (row < M && aK < K) ? A[ao + int(row) * ars + int(aK) * acs] : 0.0;
        Bs[tid.y][tid.x] = (bK < K && col < N) ? B[bo + int(bK) * brs + int(col) * bcs] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = 0; e < TILE; ++e) acc += As[tid.y][e] * Bs[e][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[co + int(row) * crs + int(col) * ccs] = acc;
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

constant uint VEC_FC [[function_constant(0)]];
constant uint GSGM_FC [[function_constant(4)]];
constant uint GSGN_FC [[function_constant(5)]];
constant uint GATM_FC [[function_constant(6)]];
constant uint GATN_FC [[function_constant(7)]];
constant uint ATTN_SPLIT_RAW [[function_constant(2)]];
constant uint ATTN_VEC_RAW [[function_constant(3)]];
constant uint ATTN_SPLIT = is_function_constant_defined(ATTN_SPLIT_RAW) ? ATTN_SPLIT_RAW : 1;
constant uint ATTN_VEC = is_function_constant_defined(ATTN_VEC_RAW) ? ATTN_VEC_RAW : 1;
constant uint NCOLG_FC [[function_constant(1)]];

kernel void gemv_t(
    device const float* X   [[buffer(0)]],
    device const float* W   [[buffer(1)]],
    device float*       OUT [[buffer(2)]],
    constant uint&      K   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant int4&      ap  [[buffer(5)]],
    constant int4&      bp  [[buffer(6)]],
    constant int&       co  [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]])
{
    const uint VEC = VEC_FC;
    const uint NC = NCOLG_FC;
    uint TG = tptg;
    uint KS = TG / NC;
    uint cg = tid % NC;
    uint ks = tid / NC;
    uint n0 = (tgid * NC + cg) * VEC;
    int xo = ap.x, xs = ap.y;
    int wo = bp.x, ws = bp.y;
    threadgroup float4 part[512];
    uint nv4 = VEC / 4;
    float4 acc[2] = {float4(0.0), float4(0.0)};
    bool vec_ok = (n0 + VEC <= N) && (ws % 4 == 0) && ((wo + int(n0)) % 4 == 0);
    if (vec_ok) {
        device const float4* W4 = (device const float4*)(W + wo + n0);
        uint ws4 = uint(ws) / 4u;
        for (uint k = ks; k < K; k += KS) {
            float xv = X[xo + int(k) * xs];
            for (uint v = 0; v < nv4; ++v) acc[v] += xv * W4[k * ws4 + v];
        }
    } else if (n0 < N) {
        for (uint k = ks; k < K; k += KS) {
            float xv = X[xo + int(k) * xs];
            for (uint j = 0; j < VEC; ++j) {
                if (n0 + j < N) acc[j / 4][j % 4] += xv * W[wo + int(k) * ws + int(n0 + j)];
            }
        }
    }
    for (uint v = 0; v < nv4; ++v) part[tid * 2 + v] = acc[v];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0 && n0 < N) {
        float4 tot[2] = {float4(0.0), float4(0.0)};
        for (uint j = 0; j < KS; ++j) {
            uint t = j * NC + cg;
            for (uint v = 0; v < nv4; ++v) tot[v] += part[t * 2 + v];
        }
        for (uint j = 0; j < VEC; ++j) {
            if (n0 + j < N) OUT[co + n0 + j] = tot[j / 4][j % 4];
        }
    }
}

kernel void gemv_sg_t(
    device const float* X   [[buffer(0)]],
    device const float* W   [[buffer(1)]],
    device float*       OUT [[buffer(2)]],
    constant uint&      K   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant int4&      ap  [[buffer(5)]],
    constant int4&      bp  [[buffer(6)]],
    constant int&       co  [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    const uint CS = NCOLG_FC;
    uint sgs = tptg / 32;
    uint cidx = lane % CS;
    uint kslice = lane / CS;
    uint KS = 32 / CS;
    uint n0 = (tgid * sgs * CS + sgid * CS + cidx) * 4;
    int xo = ap.x, xs = ap.y;
    int wo = bp.x, ws = bp.y;
    float4 acc = float4(0.0);
    bool vec_ok = (n0 + 4 <= N) && (ws % 4 == 0) && ((wo + int(n0)) % 4 == 0);
    if (vec_ok) {
        device const float4* W4 = (device const float4*)(W + wo + n0);
        uint ws4 = uint(ws) / 4u;
        for (uint k = kslice; k < K; k += KS) {
            acc += X[xo + int(k) * xs] * W4[k * ws4];
        }
    } else if (n0 < N) {
        for (uint k = kslice; k < K; k += KS) {
            float xv = X[xo + int(k) * xs];
            for (uint j = 0; j < 4; ++j) {
                if (n0 + j < N) acc[j] += xv * W[wo + int(k) * ws + int(n0 + j)];
            }
        }
    }
    for (uint sh = 16; sh >= CS; sh /= 2) {
        acc += simd_shuffle_down(acc, sh);
    }
    if (kslice == 0 && n0 < N) {
        for (uint j = 0; j < 4; ++j) {
            if (n0 + j < N) OUT[co + n0 + j] = acc[j];
        }
    }
}

kernel void gemv_rows_t(
    device const float* X   [[buffer(0)]],
    device const float* W   [[buffer(1)]],
    device float*       OUT [[buffer(2)]],
    constant uint&      K   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant int4&      ap  [[buffer(5)]],
    constant int4&      bp  [[buffer(6)]],
    constant int&       co  [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tptg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint VEC = VEC_FC;
    uint rptg = tptg / 32;
    uint r = tgid * rptg + tid / 32;
    if (r >= N) return;
    int xo = ap.x, xs = ap.y;
    int wo = bp.x, rs = bp.z;
    long base = wo + (long)r * rs;
    float acc = 0.0;
    bool vec_ok = (VEC == 4) && (base % 4 == 0) && (K % 4 == 0) && (xs == 1);
    if (vec_ok) {
        device const float4* W4 = (device const float4*)(W + base);
        device const float4* X4 = (device const float4*)(X + xo);
        for (uint k4 = lane; k4 < K / 4; k4 += 32) {
            float4 w = W4[k4];
            float4 x = X4[k4];
            acc += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
        }
    } else {
        for (uint k = lane; k < K; k += 32) {
            acc += X[xo + int(k) * xs] * W[base + k];
        }
    }
    float total = simd_sum(acc);
    if (lane == 0) OUT[co + r] = total;
}

kernel void gemv_n4(
    device const float* X   [[buffer(0)]],
    device const float* W   [[buffer(1)]],
    device float*       OUT [[buffer(2)]],
    constant uint&      K   [[buffer(3)]],
    constant uint&      N   [[buffer(4)]],
    constant int4&      ap  [[buffer(5)]],
    constant int4&      bp  [[buffer(6)]],
    constant int&       co  [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup float4 partial[256];
    uint c4 = tid % 8;
    uint ks = tid / 8;
    uint n = (tgid * 8 + c4) * 4;
    int xo = ap.x, xs = ap.y;
    int wo = bp.x, ws = bp.y;
    float4 acc = float4(0.0);
    if (n < N) {
        bool vec = (n + 4 <= N) && (ws % 4 == 0) && ((wo + int(n)) % 4 == 0);
        if (vec) {
            device const float4* W4 = (device const float4*)(W + wo + n);
            uint ws4 = uint(ws) / 4u;
            for (uint k = ks; k < K; k += 32) {
                acc += X[xo + int(k) * xs] * W4[k * ws4];
            }
        } else {
            for (uint k = ks; k < K; k += 32) {
                float xv = X[xo + int(k) * xs];
                for (uint j = 0; j < 4; ++j) {
                    if (n + j < N) acc[j] += xv * W[wo + int(k) * ws + int(n + j)];
                }
            }
        }
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks == 0 && n < N) {
        float4 total = float4(0.0);
        for (uint j = 0; j < 32; ++j) total += partial[c4 + 8 * j];
        for (uint j = 0; j < 4; ++j) {
            if (n + j < N) OUT[co + n + j] = total[j];
        }
    }
}

static inline long xaddr(
    constant int* m, constant int* spec,
    constant int* im0, constant int* im1, constant int* im2, constant int* im3,
    device const int* I0, device const int* I1, device const int* I2, device const int* I3,
    uint i)
{
    long off = strided_off(m, i);
    int nind = spec[0];
    for (int k = 0; k < nind; ++k) {
        int b = spec[1 + 2 * k];
        long st = spec[2 + 2 * k];
        int v;
        if (b == 0) v = I0[strided_off(im0, i)];
        else if (b == 1) v = I1[strided_off(im1, i)];
        else if (b == 2) v = I2[strided_off(im2, i)];
        else v = I3[strided_off(im3, i)];
        off += (long)v * st;
    }
    return off;
}

#define XCOPY_PARAMS \
    device const int* I0 [[buffer(2)]], \
    device const int* I1 [[buffer(3)]], \
    device const int* I2 [[buffer(4)]], \
    device const int* I3 [[buffer(5)]], \
    constant uint& n     [[buffer(6)]], \
    constant int* rmeta  [[buffer(7)]], \
    constant int* wmeta  [[buffer(8)]], \
    constant int* rspec  [[buffer(9)]], \
    constant int* wspec  [[buffer(10)]], \
    constant int* im0    [[buffer(11)]], \
    constant int* im1    [[buffer(12)]], \
    constant int* im2    [[buffer(13)]], \
    constant int* im3    [[buffer(14)]], \
    uint gid [[thread_position_in_grid]]

kernel void xcopy_f(
    device const float* SRC [[buffer(0)]],
    device float*       DST [[buffer(1)]],
    XCOPY_PARAMS)
{
    if (gid >= n) return;
    long r = xaddr(rmeta, rspec, im0, im1, im2, im3, I0, I1, I2, I3, gid);
    long w = xaddr(wmeta, wspec, im0, im1, im2, im3, I0, I1, I2, I3, gid);
    DST[w] = SRC[r];
}

kernel void xcopy_i(
    device const int* SRC [[buffer(0)]],
    device int*       DST [[buffer(1)]],
    XCOPY_PARAMS)
{
    if (gid >= n) return;
    long r = xaddr(rmeta, rspec, im0, im1, im2, im3, I0, I1, I2, I3, gid);
    long w = xaddr(wmeta, wspec, im0, im1, im2, im3, I0, I1, I2, I3, gid);
    DST[w] = SRC[r];
}

kernel void xcopy_i2f(
    device const int* SRC [[buffer(0)]],
    device float*     DST [[buffer(1)]],
    XCOPY_PARAMS)
{
    if (gid >= n) return;
    long r = xaddr(rmeta, rspec, im0, im1, im2, im3, I0, I1, I2, I3, gid);
    long w = xaddr(wmeta, wspec, im0, im1, im2, im3, I0, I1, I2, I3, gid);
    DST[w] = float(SRC[r]);
}

kernel void lt_strided(
    device const int*     A [[buffer(0)]],
    device const int*     B [[buffer(1)]],
    device unsigned char* C [[buffer(2)]],
    constant uint&        n [[buffer(3)]],
    constant int*        ma [[buffer(4)]],
    constant int*        mb [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    C[gid] = A[strided_off(ma, gid)] < B[strided_off(mb, gid)] ? 1 : 0;
}

kernel void reduce_sum_l(
    device const float* X    [[buffer(0)]],
    device float*       OUT  [[buffer(1)]],
    constant uint&      outer [[buffer(2)]],
    constant uint&      axis  [[buffer(3)]],
    constant uint&      inner [[buffer(4)]],
    constant int*       m     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= outer * inner) return;
    uint o = gid / inner;
    uint in_ = gid % inner;
    float s = 0.0;
    for (uint a = 0; a < axis; ++a) {
        s += X[strided_off(m, (o * axis + a) * inner + in_)];
    }
    OUT[gid] = s;
}

kernel void reduce_max_l(
    device const float* X    [[buffer(0)]],
    device float*       OUT  [[buffer(1)]],
    constant uint&      outer [[buffer(2)]],
    constant uint&      axis  [[buffer(3)]],
    constant uint&      inner [[buffer(4)]],
    constant int*       m     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= outer * inner) return;
    uint o = gid / inner;
    uint in_ = gid % inner;
    float v = -INFINITY;
    for (uint a = 0; a < axis; ++a) {
        v = max(v, X[strided_off(m, (o * axis + a) * inner + in_)]);
    }
    OUT[gid] = v;
}

kernel void ew_chain(
    device const float* E0  [[buffer(0)]],
    device const float* E1  [[buffer(1)]],
    device const float* E2  [[buffer(2)]],
    device const float* E3  [[buffer(3)]],
    device float*       OUT [[buffer(4)]],
    constant uint&      n   [[buffer(5)]],
    constant int*       prog [[buffer(6)]],
    constant int*       m0  [[buffer(7)]],
    constant int*       m1  [[buffer(8)]],
    constant int*       m2  [[buffer(9)]],
    constant int*       m3  [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = E0[strided_off(m0, gid)];
    int steps = prog[0];
    for (int s = 0; s < steps; ++s) {
        int op = prog[1 + 3 * s];
        int kind = prog[2 + 3 * s];
        int payload = prog[3 + 3 * s];
        float a = 0.0;
        if (kind == 1) {
            a = as_type<float>(payload);
        } else if (kind == 2) {
            if (payload == 1) a = E1[strided_off(m1, gid)];
            else if (payload == 2) a = E2[strided_off(m2, gid)];
            else a = E3[strided_off(m3, gid)];
        }
        switch (op) {
            case 0: v = v * a; break;
            case 1: v = v + a; break;
            case 2: v = 1.0f / v; break;
            case 3: v = sqrt(v); break;
            case 4: v = exp2(v); break;
        }
    }
    OUT[gid] = v;
}

kernel void gemm_sg(
    device const float* A    [[buffer(0)]],
    device const float* B    [[buffer(1)]],
    device float*       C    [[buffer(2)]],
    constant int4&      dims [[buffer(3)]],
    constant int4&      ap   [[buffer(4)]],
    constant int4&      bp   [[buffer(5)]],
    constant int4&      cp   [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    int k = dims.y;
    int sgm = int(sgid) / int(GSGN_FC);
    int sgn = int(sgid) % int(GSGN_FC);
    int i0 = int(tgid.y) * int(GSGM_FC * GATM_FC * 8u) + sgm * int(GATM_FC) * 8;
    int j0 = int(tgid.x) * int(GSGN_FC * GATN_FC * 8u) + sgn * int(GATN_FC) * 8;
    long am = ap.y, ak = ap.z;
    long bk = bp.y, bn = bp.z;
    bool at = ak != 1;
    bool bt = bn != 1;
    ulong lda = ulong(at ? ak : am);
    ulong ldb = ulong(bt ? bn : bk);
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < GATM_FC; ++i)
        for (uint j = 0; j < GATN_FC; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int kk = 0; kk < k; kk += 8) {
        simdgroup_float8x8 af[4];
        simdgroup_float8x8 bf[4];
        for (uint i = 0; i < GATM_FC; ++i) {
            long base = long(ap.x) + (long(i0) + long(i) * 8) * am + long(kk) * ak;
            simdgroup_load(af[i], A + base, lda, ulong2(0, 0), at);
        }
        for (uint j = 0; j < GATN_FC; ++j) {
            long base = long(bp.x) + long(kk) * bk + (long(j0) + long(j) * 8) * bn;
            simdgroup_load(bf[j], B + base, ldb, ulong2(0, 0), bt);
        }
        for (uint i = 0; i < GATM_FC; ++i)
            for (uint j = 0; j < GATN_FC; ++j)
                simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    long cm = cp.y;
    for (uint i = 0; i < GATM_FC; ++i)
        for (uint j = 0; j < GATN_FC; ++j) {
            long base = long(cp.x) + (long(i0) + long(i) * 8) * cm + long(j0) + long(j) * 8;
            simdgroup_store(acc[i][j], C + base, ulong(cm), ulong2(0, 0));
        }
}

kernel void attn_online(
    device const float* Q    [[buffer(0)]],
    device const float* Kc   [[buffer(1)]],
    device const float* Vc   [[buffer(2)]],
    device const float* BIAS [[buffer(3)]],
    device float*       OUT  [[buffer(4)]],
    constant int*       p    [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    int KV = p[0], G = p[1], S = p[2], CTX = p[3], HD = p[4];
    uint jptg = 8 / ATTN_SPLIT;
    int job = int(tgid * jptg + sgid / ATTN_SPLIT);
    uint sgj = sgid % ATTN_SPLIT;
    int total = KV * G * S;
    bool valid = job < total;
    float m = -INFINITY;
    float l = 0.0;
    float acc0 = 0.0;
    float acc1 = 0.0;
    int e0 = int(lane);
    int e1 = int(lane) + 32;
    float scale = as_type<float>(p[23]);
    if (valid) {
        int s = job % S;
        int head = job / S;
        int kv = head / G;
        int g = head % G;
        int qoff = p[5] + kv * p[6] + g * p[7] + s * p[8];
        int koff = p[10] + kv * p[11];
        int voff = p[14] + kv * p[15];
        int boff = p[18] + kv * p[19] + g * p[20] + s * p[21];
        int chunk = (CTX + int(ATTN_SPLIT) - 1) / int(ATTN_SPLIT);
        int c0 = int(sgj) * chunk;
        int cend = min(CTX, c0 + chunk);
        for (int c = c0; c < cend; ++c) {
            float part = 0.0;
            if (ATTN_VEC == 4) {
                device const float4* Q4 = (device const float4*)(Q + qoff);
                device const float4* K4 = (device const float4*)(Kc + koff + c * p[12]);
                for (int e4 = int(lane); e4 < HD / 4; e4 += 32) {
                    float4 qv = Q4[e4];
                    float4 kw = K4[e4];
                    part += qv.x * kw.x + qv.y * kw.y + qv.z * kw.z + qv.w * kw.w;
                }
            } else {
                for (int e = int(lane); e < HD; e += 32) {
                    part += Q[qoff + e * p[9]] * Kc[koff + c * p[12] + e * p[13]];
                }
            }
            float dot = simd_sum(part);
            float x = dot * scale + BIAS[boff + c * p[22]];
            float mn = max(m, x);
            float corr = exp(m - mn);
            float pr = exp(x - mn);
            l = l * corr + pr;
            float v0 = e0 < HD ? Vc[voff + c * p[16] + e0 * p[17]] : 0.0;
            float v1 = e1 < HD ? Vc[voff + c * p[16] + e1 * p[17]] : 0.0;
            acc0 = acc0 * corr + pr * v0;
            acc1 = acc1 * corr + pr * v1;
            m = mn;
        }
    }
    if (ATTN_SPLIT == 1) {
        if (!valid) return;
        float inv = 1.0f / l;
        int ob = job * HD;
        if (e0 < HD) OUT[ob + e0] = acc0 * inv;
        if (e1 < HD) OUT[ob + e1] = acc1 * inv;
        return;
    }
    threadgroup float ms[8];
    threadgroup float ls[8];
    threadgroup float accs[8 * 64];
    if (lane == 0) {
        ms[sgid] = m;
        ls[sgid] = l;
    }
    if (e0 < HD) accs[sgid * 64 + e0] = acc0;
    if (e1 < HD) accs[sgid * 64 + e1] = acc1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid || sgj != 0) return;
    float M = -INFINITY;
    for (uint i = 0; i < ATTN_SPLIT; ++i) M = max(M, ms[sgid + i]);
    float L = 0.0;
    float o0 = 0.0;
    float o1 = 0.0;
    for (uint i = 0; i < ATTN_SPLIT; ++i) {
        float w = exp(ms[sgid + i] - M);
        L += ls[sgid + i] * w;
        if (e0 < HD) o0 += accs[(sgid + i) * 64 + e0] * w;
        if (e1 < HD) o1 += accs[(sgid + i) * 64 + e1] * w;
    }
    float inv = 1.0f / L;
    int ob = job * HD;
    if (e0 < HD) OUT[ob + e0] = o0 * inv;
    if (e1 < HD) OUT[ob + e1] = o1 * inv;
}

kernel void attn_decode(
    device const float* Q    [[buffer(0)]],
    device const float* Kc   [[buffer(1)]],
    device const float* Vc   [[buffer(2)]],
    device const float* BIAS [[buffer(3)]],
    device float*       OUT  [[buffer(4)]],
    constant int*       p    [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    int KV = p[0], G = p[1], CTX = p[2], HD = p[3];
    int head = int(tgid) * int((256u) / 32u) + int(sgid);
    if (head >= KV * G) return;
    int kv = head / G;
    int g = head % G;
    int qoff = p[4] + kv * p[5] + g * p[6];
    int koff = p[8] + kv * p[9];
    int voff = p[12] + kv * p[13];
    int boff = p[16] + kv * p[17] + g * p[18];
    float scale = as_type<float>(p[20]);
    float scores[8];
    for (int c = 0; c < CTX; ++c) {
        float acc = 0.0;
        for (int e = int(lane); e < HD; e += 32) {
            acc += Q[qoff + e * p[7]] * Kc[koff + c * p[10] + e * p[11]];
        }
        float dot = simd_sum(acc);
        scores[c] = dot * scale + BIAS[boff + c * p[19]];
    }
    float m = -INFINITY;
    for (int c = 0; c < CTX; ++c) m = max(m, scores[c]);
    float ssum = 0.0;
    for (int c = 0; c < CTX; ++c) {
        scores[c] = exp(scores[c] - m);
        ssum += scores[c];
    }
    float inv = 1.0f / ssum;
    for (int e = int(lane); e < HD; e += 32) {
        float o = 0.0;
        for (int c = 0; c < CTX; ++c) {
            o += scores[c] * inv * Vc[voff + c * p[14] + e * p[15]];
        }
        OUT[head * HD + e] = o;
    }
}

kernel void rmsnorm_row(
    device const float* X   [[buffer(0)]],
    device float*       OUT [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      cols [[buffer(3)]],
    constant float&     eps  [[buffer(4)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup float partial[256];
    if (tgid >= rows) return;
    device const float* x = X + tgid * cols;
    float acc = 0.0;
    for (uint j = tid; j < cols; j += 256) {
        float v = x[j];
        acc += v * v;
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0;
        for (uint j = 0; j < 256; ++j) s += partial[j];
        partial[0] = 1.0f / sqrt(s * (1.0f / float(cols)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = partial[0];
    device float* o = OUT + tgid * cols;
    for (uint j = tid; j < cols; j += 256) {
        o[j] = x[j] * r;
    }
}

kernel void silu_mul(
    device const float* G   [[buffer(0)]],
    device const float* U   [[buffer(1)]],
    device float*       OUT [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    constant float&     kk  [[buffer(4)]],
    constant float&     c3  [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = G[gid];
    float sig = 1.0f / (exp2(g * kk) + c3);
    OUT[gid] = g * sig * U[gid];
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