use crate::kernels::{KERNELS, KERNEL_SOURCE, TILE};
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::ffi::c_void;

pub struct CuteMetal {
    device: Device,
    queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

impl CuteMetal {
    pub fn new() -> Self {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let opts = metal::CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(KERNEL_SOURCE, &opts)
            .expect("compile MSL");
        let pipelines = KERNELS
            .iter()
            .map(|(name, _)| {
                let f = lib.get_function(name, None).expect("get function");
                let p = device
                    .new_compute_pipeline_state_with_function(&f)
                    .expect("pipeline");
                (*name, p)
            })
            .collect();
        Self { device, queue, pipelines }
    }

    fn pipe(&self, name: &str) -> &ComputePipelineState {
        &self.pipelines[name]
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

    pub fn command_buffer(&self) -> &metal::CommandBufferRef {
        self.queue.new_command_buffer()
    }

    #[allow(clippy::too_many_arguments)]
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
        enc.set_compute_pipeline_state(self.pipe("tiled_matmul"));
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
        enc.set_compute_pipeline_state(self.pipe("materialize"));
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

    pub fn gather_into(
        &self,
        cb: &metal::CommandBufferRef,
        src: &Buffer,
        dst: &Buffer,
        ext: [u32; 4],
        st: [i32; 4],
    ) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipe("gather3"));
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

    pub fn relu_into(&self, cb: &metal::CommandBufferRef, x: &Buffer, n: usize) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipe("relu"));
        enc.set_buffer(0, Some(x), 0);
        let nu = n as u32;
        enc.set_bytes(1, 4, &nu as *const u32 as *const c_void);
        let tg = MTLSize::new(256, 1, 1);
        enc.dispatch_thread_groups(MTLSize::new((n as u64).div_ceil(256), 1, 1), tg);
        enc.end_encoding();
    }

    pub fn add_into(
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
        enc.set_compute_pipeline_state(self.pipe("add2"));
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

    pub fn softmax_into(&self, cb: &metal::CommandBufferRef, x: &Buffer, rows: usize, cols: usize) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(self.pipe("softmax_rows"));
        enc.set_buffer(0, Some(x), 0);
        let (r, co) = (rows as u32, cols as u32);
        enc.set_bytes(1, 4, &r as *const u32 as *const c_void);
        enc.set_bytes(2, 4, &co as *const u32 as *const c_void);
        let tg = MTLSize::new(64, 1, 1);
        enc.dispatch_thread_groups(MTLSize::new((rows as u64).div_ceil(64), 1, 1), tg);
        enc.end_encoding();
    }
}
