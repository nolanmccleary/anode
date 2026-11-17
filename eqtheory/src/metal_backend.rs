use crate::kernels::{KERNELS, KERNEL_SOURCE, TILE};
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::ffi::c_void;

pub struct CuteMetal {
    device: Device,
    queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    lib: metal::Library,
    gemv_variants: std::cell::RefCell<HashMap<(u32, u32), ComputePipelineState>>,
    gemv_choice: std::cell::RefCell<HashMap<(usize, usize, bool), (u32, u32, u64)>>,
    rows_variants: std::cell::RefCell<HashMap<u32, ComputePipelineState>>,
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
        let me = Self {
            device,
            queue,
            pipelines,
            lib,
            gemv_variants: Default::default(),
            gemv_choice: Default::default(),
            rows_variants: Default::default(),
        };
        let _ = me.gemv_pipeline(4, 8);
        let _ = me.gemv_rows_pipeline(4);
        me
    }

    pub fn gemv_rows_pipeline(&self, vec: u32) -> ComputePipelineState {
        if let Some(p) = self.rows_variants.borrow().get(&vec) {
            return p.clone();
        }
        let fcv = metal::FunctionConstantValues::new();
        fcv.set_constant_value_at_index(
            &vec as *const u32 as *const c_void,
            metal::MTLDataType::UInt,
            0,
        );
        let one = 1u32;
        fcv.set_constant_value_at_index(
            &one as *const u32 as *const c_void,
            metal::MTLDataType::UInt,
            1,
        );
        let f = self.lib.get_function("gemv_rows_t", Some(fcv)).expect("gemv_rows_t");
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .expect("gemv_rows_t pipeline");
        self.rows_variants.borrow_mut().insert(vec, p.clone());
        p
    }

    fn encode_gemv_rows(
        &self,
        cb: &metal::CommandBufferRef,
        pipe: &ComputePipelineState,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        k: usize,
        n: usize,
        xap: [i32; 4],
        wbp: [i32; 4],
        co: i32,
        tg: u64,
    ) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        let (ku, nu) = (k as u32, n as u32);
        enc.set_bytes(3, 4, &ku as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
        enc.set_bytes(5, 16, xap.as_ptr() as *const c_void);
        enc.set_bytes(6, 16, wbp.as_ptr() as *const c_void);
        enc.set_bytes(7, 4, &co as *const i32 as *const c_void);
        let rptg = tg / 32;
        let tgs = (n as u64).div_ceil(rptg);
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
    }

    pub fn gemv_pipeline(&self, vec: u32, ncolg: u32) -> ComputePipelineState {
        if let Some(p) = self.gemv_variants.borrow().get(&(vec, ncolg)) {
            return p.clone();
        }
        let fcv = metal::FunctionConstantValues::new();
        fcv.set_constant_value_at_index(
            &vec as *const u32 as *const c_void,
            metal::MTLDataType::UInt,
            0,
        );
        fcv.set_constant_value_at_index(
            &ncolg as *const u32 as *const c_void,
            metal::MTLDataType::UInt,
            1,
        );
        let f = self.lib.get_function("gemv_t", Some(fcv)).expect("gemv_t specialize");
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .expect("gemv_t pipeline");
        self.gemv_variants.borrow_mut().insert((vec, ncolg), p.clone());
        p
    }

    fn encode_gemv_t(
        &self,
        cb: &metal::CommandBufferRef,
        pipe: &ComputePipelineState,
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        k: usize,
        n: usize,
        xap: [i32; 4],
        wbp: [i32; 4],
        co: i32,
        vec: u32,
        ncolg: u32,
        tg: u64,
    ) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        let (ku, nu) = (k as u32, n as u32);
        enc.set_bytes(3, 4, &ku as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
        enc.set_bytes(5, 16, xap.as_ptr() as *const c_void);
        enc.set_bytes(6, 16, wbp.as_ptr() as *const c_void);
        enc.set_bytes(7, 4, &co as *const i32 as *const c_void);
        let cols_per_tg = (vec * ncolg) as u64;
        let tgs = (n as u64).div_ceil(cols_per_tg);
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
    }

    pub fn tune_gemv(&self, k: usize, n: usize, a: &Buffer, _b: &Buffer, ap: [i32; 4], bp: [i32; 4]) {
        let col = bp[2] == 1;
        let rows = !col && bp[1] == 1;
        if !col && !rows {
            return;
        }
        let key = (k, n, col);
        if self.gemv_choice.borrow().contains_key(&key) {
            return;
        }
        if std::env::var("GEMV_NOTUNE").is_ok() {
            return;
        }
        let bytes = k * n * 4;
        let copies = (96_000_000usize / bytes).clamp(1, 8);
        let wscratch = self.alloc_bytes(bytes * copies);
        let out_scratch = self.alloc_bytes(n * 4);
        let xap = [0, ap[2], 0, 0];
        let mut best = (f64::INFINITY, (4u32, 8u32, 256u64));
        let candidates: Vec<(u32, u32, u64)> = if rows {
            let mut v = Vec::new();
            for tg in [128u64, 256u64] {
                for vec in [4u32] {
                    v.push((vec, 1, tg));
                }
            }
            v
        } else {
            let mut v = Vec::new();
            for tg in [128u64, 256u64] {
                for ncolg in [4u32, 8u32, 16u32] {
                    let ks = tg / ncolg as u64;
                    if !(8..=64).contains(&ks) {
                        continue;
                    }
                    for vec in [4u32, 8u32] {
                        v.push((vec, ncolg, tg));
                    }
                }
            }
            v
        };
        for (vec, ncolg, tg) in candidates {
            let run = |iters: usize, offs: usize| {
                let cb = self.queue.new_command_buffer();
                for it in 0..iters {
                    let off = (((it + offs) % copies) * bytes) as u64;
                    let enc_w = &wscratch;
                    let wbp0 = if rows {
                        [off as i32 / 4, 1, bp[2], 0]
                    } else {
                        [off as i32 / 4, bp[1], 1, 0]
                    };
                    if rows {
                        let pipe = self.gemv_rows_pipeline(vec);
                        self.encode_gemv_rows(cb, &pipe, a, enc_w, &out_scratch, k, n, xap, wbp0, 0, tg);
                    } else {
                        let pipe = self.gemv_pipeline(vec, ncolg);
                        self.encode_gemv_t(cb, &pipe, a, enc_w, &out_scratch, k, n, xap, wbp0, 0, vec, ncolg, tg);
                    }
                }
                cb.commit();
                cb.wait_until_completed();
            };
            run(2, 0);
            let t = std::time::Instant::now();
            run(4, 2);
            let ms = t.elapsed().as_secs_f64() * 1e3 / 4.0;
            if ms < best.0 {
                best = (ms, (vec, ncolg, tg));
            }
        }
        if std::env::var("GEMVTUNE").is_ok() {
            eprintln!(
                "gemv tune K={k} N={n} rows={rows}: vec={} ncolg={} tg={} ({:.3} ms, {:.1} GB/s)",
                best.1.0,
                best.1.1,
                best.1.2,
                best.0,
                bytes as f64 / (best.0 / 1e3) / 1e9
            );
        }
        self.gemv_choice.borrow_mut().insert(key, best.1);
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

    pub fn owned_command_buffer(&self) -> metal::CommandBuffer {
        self.queue.new_command_buffer().to_owned()
    }

    pub fn external_pipeline(
        &self,
        source: &str,
        fast_math: bool,
        name: &str,
    ) -> ComputePipelineState {
        let opts = metal::CompileOptions::new();
        opts.set_fast_math_enabled(fast_math);
        let lib = self
            .device
            .new_library_with_source(source, &opts)
            .expect("compile external MSL");
        let f = lib.get_function(name, None).expect("external function");
        self.device
            .new_compute_pipeline_state_with_function(&f)
            .expect("external pipeline")
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.pipe(name)
    }

    pub fn upload_i32(&self, data: &[i32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            (data.len() * 4).max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn upload_u8(&self, data: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            data.len().max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn alloc_bytes(&self, bytes: usize) -> Buffer {
        self.device
            .new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared)
    }

    pub fn read_i32(&self, buf: &Buffer, n: usize) -> Vec<i32> {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const i32, n) }.to_vec()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_or_gemv_into(
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
        if m == 1 && bp[2] == 1 && cp[2] == 1 {
            let choice = self.gemv_choice.borrow().get(&(k, n, true)).copied();
            let (vec, ncolg, tg) = choice.unwrap_or((4, 8, 256));
            let pipe = self.gemv_pipeline(vec, ncolg);
            let xap = [ap[0], ap[2], 0, 0];
            let wbp = [bp[0], bp[1], 0, 0];
            self.encode_gemv_t(cb, &pipe, a, b, c, k, n, xap, wbp, cp[0], vec, ncolg, tg);
        } else if m == 1 && bp[1] == 1 && cp[2] == 1 {
            let choice = self.gemv_choice.borrow().get(&(k, n, false)).copied();
            let (vec, _, tg) = choice.unwrap_or((4, 1, 256));
            let pipe = self.gemv_rows_pipeline(vec);
            let xap = [ap[0], ap[2], 0, 0];
            let wbp = [bp[0], 1, bp[2], 0];
            self.encode_gemv_rows(cb, &pipe, a, b, c, k, n, xap, wbp, cp[0], tg);
        } else {
            self.matmul_into(cb, a, b, c, m, k, n, ap, bp, cp);
        }
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
