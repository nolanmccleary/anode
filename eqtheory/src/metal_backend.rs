use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;

const TILE: u64 = 16;

const SRC: &str = r#"
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

kernel void add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
    constant uint&      n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    C[gid] = A[gid] + B[gid];
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

pub struct CuteMetal {
    device: Device,
    queue: CommandQueue,
    matmul: ComputePipelineState,
    materialize: ComputePipelineState,
    gather: ComputePipelineState,
    relu: ComputePipelineState,
    add: ComputePipelineState,
    add2: ComputePipelineState,
    softmax: ComputePipelineState,
}

impl CuteMetal {
    pub fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let opts = metal::CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(SRC, &opts)
            .expect("compile MSL");
        let make = |name: &str| {
            let f = lib.get_function(name, None).expect("get function");
            device
                .new_compute_pipeline_state_with_function(&f)
                .expect("pipeline")
        };
        let matmul = make("tiled_matmul");
        let materialize = make("materialize");
        let gather = make("gather3");
        let relu = make("relu");
        let add = make("add");
        let add2 = make("add2");
        let softmax = make("softmax_rows");
        Self { device, queue, matmul, materialize, gather, relu, add, add2, softmax }
    }

    pub fn add2_into(
        &self,
        cb: &metal::CommandBufferRef,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        rows: usize,
        cols: usize,
        bp: [i32; 4],
    ) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.add2);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        let (r, co) = (rows as u32, cols as u32);
        enc.set_bytes(3, 4, &r as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &co as *const u32 as *const c_void);
        enc.set_bytes(5, 16, bp.as_ptr() as *const c_void);
        let tg = MTLSize::new(TILE, TILE, 1);
        let grid = MTLSize::new((cols as u64).div_ceil(TILE), (rows as u64).div_ceil(TILE), 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    pub fn relu_into(&self, cb: &metal::CommandBufferRef, x: &Buffer, n: usize) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.relu);
        enc.set_buffer(0, Some(x), 0);
        let nu = n as u32;
        enc.set_bytes(1, 4, &nu as *const u32 as *const c_void);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(256), 1, 1), tg);
        enc.end_encoding();
    }

    pub fn add_into(&self, cb: &metal::CommandBufferRef, a: &Buffer, b: &Buffer, c: &Buffer, n: usize) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.add);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        let nu = n as u32;
        enc.set_bytes(3, 4, &nu as *const u32 as *const c_void);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(256), 1, 1), tg);
        enc.end_encoding();
    }

    pub fn upload(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            (data.len() * 4).max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn alloc(&self, n: usize) -> Buffer {
        self.device
            .new_buffer((n * 4).max(4) as u64, MTLResourceOptions::StorageModeShared)
    }

    pub fn read(&self, buf: &Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }.to_vec()
    }

    pub fn matmul_into(
        &self,
        cb: &metal::CommandBufferRef,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        m: usize,
        k: usize,
        n: usize,
        ap: [i32; 4],
        bp: [i32; 4],
        cp: [i32; 4],
    ) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.matmul);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);
        enc.set_bytes(3, 4, &mu as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &ku as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &nu as *const u32 as *const c_void);
        enc.set_bytes(6, 16, ap.as_ptr() as *const c_void);
        enc.set_bytes(7, 16, bp.as_ptr() as *const c_void);
        enc.set_bytes(8, 16, cp.as_ptr() as *const c_void);
        let tg = MTLSize::new(TILE, TILE, 1);
        let grid = MTLSize::new((n as u64).div_ceil(TILE), (m as u64).div_ceil(TILE), 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    pub fn gather_into(
        &self,
        cb: &metal::CommandBufferRef,
        src: &Buffer,
        dst: &Buffer,
        ext: [u32; 4],
        st: [i32; 4],
    ) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.gather);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_bytes(2, 16, ext.as_ptr() as *const c_void);
        enc.set_bytes(3, 16, st.as_ptr() as *const c_void);
        let total = (ext[0] * ext[1] * ext[2]) as u64;
        let tg = MTLSize::new(256, 1, 1);
        let grid = MTLSize::new(total.div_ceil(256), 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    pub fn materialize_into(
        &self,
        cb: &metal::CommandBufferRef,
        src: &Buffer,
        dst: &Buffer,
        rows: usize,
        cols: usize,
        sp: [i32; 4],
    ) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.materialize);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        let (r, c) = (rows as u32, cols as u32);
        enc.set_bytes(2, 4, &r as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &c as *const u32 as *const c_void);
        enc.set_bytes(4, 16, sp.as_ptr() as *const c_void);
        let tg = MTLSize::new(TILE, TILE, 1);
        let grid = MTLSize::new((cols as u64).div_ceil(TILE), (rows as u64).div_ceil(TILE), 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    pub fn softmax_into(
        &self,
        cb: &metal::CommandBufferRef,
        x: &Buffer,
        rows: usize,
        cols: usize,
    ) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.softmax);
        enc.set_buffer(0, Some(x), 0);
        let (r, co) = (rows as u32, cols as u32);
        enc.set_bytes(1, 4, &r as *const u32 as *const c_void);
        enc.set_bytes(2, 4, &co as *const u32 as *const c_void);
        let tg = MTLSize::new(64, 1, 1);
        let grid = MTLSize::new(rows.div_ceil(64) as u64, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
    }

    pub fn command_buffer(&self) -> &metal::CommandBufferRef {
        self.queue.new_command_buffer()
    }
}
