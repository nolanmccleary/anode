use crate::kernels::{KERNELS, KERNEL_SOURCE, TILE};
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::ffi::c_void;

const PREAMBLE: &str = "#include <metal_stdlib>\nusing namespace metal;\nstatic inline long strided_off(constant int* m, uint i){ long off=m[1]; long r=i; int nm=m[0]; for(int k=0;k<nm;++k){ long sz=m[2+k]; off += (r % sz) * (long)m[10+k]; r/=sz; } return off; }";

pub struct CuteMetal {
    device: Device,
    queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    lib: metal::Library,
    gemv_variants: std::cell::RefCell<HashMap<(u32, u32), ComputePipelineState>>,
    rows_variants: std::cell::RefCell<HashMap<u32, ComputePipelineState>>,
    attn_variants: std::cell::RefCell<HashMap<(u32, u32), ComputePipelineState>>,
    gemm_variants: std::cell::RefCell<HashMap<(u32, u32, u32, u32, u32), ComputePipelineState>>,
    sample_ctx: std::cell::RefCell<Option<SampleCtx>>,
    kernels: Vec<Box<dyn crate::kernelset::Kernel>>,
    ksearch: std::cell::RefCell<HashMap<Vec<i64>, (usize, crate::kernelset::Variant)>>,
}

pub struct SampleCtx {
    pub buf: metal::CounterSampleBuffer,
    pub labels: Vec<String>,
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
                let f = if *name == "attn_online" {
                    let fcv = metal::FunctionConstantValues::new();
                    let one = 1u32;
                    fcv.set_constant_value_at_index(&one as *const u32 as *const c_void, metal::MTLDataType::UInt, 2);
                    fcv.set_constant_value_at_index(&one as *const u32 as *const c_void, metal::MTLDataType::UInt, 3);
                    lib.get_function(name, Some(fcv)).expect("get function")
                } else {
                    lib.get_function(name, None).expect("get function")
                };
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
            rows_variants: Default::default(),
            attn_variants: Default::default(),
            gemm_variants: Default::default(),
            sample_ctx: Default::default(),
            kernels: crate::kernelset::registry(),
            ksearch: Default::default(),
        };
        let _ = me.gemv_pipeline(4, 8);
        let _ = me.gemv_rows_pipeline(4);
        me
    }

    pub fn gemm_pipeline(&self, v: (u32, u32, u32, u32, u32)) -> ComputePipelineState {
        if let Some(p) = self.gemm_variants.borrow().get(&v) {
            return p.clone();
        }
        let fcv = metal::FunctionConstantValues::new();
        for (i, val) in [v.0, v.1, v.2, v.3].iter().enumerate() {
            fcv.set_constant_value_at_index(
                val as *const u32 as *const c_void,
                metal::MTLDataType::UInt,
                (4 + i) as u64,
            );
        }
        fcv.set_constant_value_at_index(
            &v.4 as *const u32 as *const c_void,
            metal::MTLDataType::UInt,
            10,
        );
        let f = self.lib.get_function("gemm_sg", Some(fcv)).expect("gemm_sg fc");
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .expect("gemm_sg pipeline");
        self.gemm_variants.borrow_mut().insert(v, p.clone());
        p
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gemm_sg(
        &self,
        cb: &metal::CommandBufferRef,
        v: (u32, u32, u32, u32, u32),
        a: &Buffer,
        b: &Buffer,
        c: &Buffer,
        m: usize,
        k: usize,
        n: usize,
        ap: [i32; 4],
        bp: [i32; 4],
        cp: [i32; 4],
        batch: &[i32],
        nbatch: i64,
    ) {
        let pipe = self.gemm_pipeline(v);
        let enc = self.enc(cb, "gemv_sg");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        let dims = [m as i32, k as i32, n as i32, 0];
        enc.set_bytes(3, 16, dims.as_ptr() as *const c_void);
        enc.set_bytes(4, 16, ap.as_ptr() as *const c_void);
        enc.set_bytes(5, 16, bp.as_ptr() as *const c_void);
        enc.set_bytes(6, 16, cp.as_ptr() as *const c_void);
        enc.set_bytes(7, (batch.len() * 4) as u64, batch.as_ptr() as *const c_void);
        let bm = (v.0 * v.2 * 8) as u64;
        let bn = (v.1 * v.3 * 8) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(n as u64 / bn, m as u64 / bm, nbatch as u64),
            MTLSize::new((32 * v.0 * v.1) as u64, 1, 1),
        );
        enc.end_encoding();
    }




    pub fn flash_pipeline(&self, sgs: u32, bk: u32) -> ComputePipelineState {
        let key = (1000 + sgs, bk);
        if let Some(p) = self.attn_variants.borrow().get(&key) {
            return p.clone();
        }
        let fcv = metal::FunctionConstantValues::new();
        fcv.set_constant_value_at_index(&sgs as *const u32 as *const c_void, metal::MTLDataType::UInt, 8);
        fcv.set_constant_value_at_index(&bk as *const u32 as *const c_void, metal::MTLDataType::UInt, 9);
        let f = self.lib.get_function("attn_flash", Some(fcv)).expect("attn_flash fc");
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .expect("attn_flash pipeline");
        self.attn_variants.borrow_mut().insert(key, p.clone());
        p
    }

    pub fn attn_pipeline(&self, split: u32, vec: u32) -> ComputePipelineState {
        let key = (split, vec);
        if let Some(p) = self.attn_variants.borrow().get(&key) {
            return p.clone();
        }
        let fcv = metal::FunctionConstantValues::new();
        fcv.set_constant_value_at_index(
            &split as *const u32 as *const c_void,
            metal::MTLDataType::UInt,
            2,
        );
        fcv.set_constant_value_at_index(
            &vec as *const u32 as *const c_void,
            metal::MTLDataType::UInt,
            3,
        );
        let f = self.lib.get_function("attn_online", Some(fcv)).expect("attn_online fc");
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .expect("attn_online pipeline");
        self.attn_variants.borrow_mut().insert(key, p.clone());
        p
    }

    pub fn attn_grid(p: &[i32; 25], kind: u32, x: u32) -> (u64, u64) {
        let jobs = (p[0] as i64 * p[1] as i64 * p[2] as i64) as u64;
        if kind == 1 {
            let heads = (p[0] * p[1]) as u64;
            let tgs = heads * (p[2] as u64 / (8 * x as u64));
            (tgs * 32 * x as u64, 32 * x as u64)
        } else {
            (jobs.div_ceil(8 / x as u64) * 256, 256)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_attn(
        &self,
        cb: &metal::CommandBufferRef,
        kind: u32,
        x: u32,
        y: u32,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        bias: &Buffer,
        out: &Buffer,
        p: &[i32; 25],
    ) {
        let pipe = if kind == 1 {
            self.flash_pipeline(x, y)
        } else {
            self.attn_pipeline(x, y)
        };
        let enc = self.enc(cb, "attn");
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k), 0);
        enc.set_buffer(2, Some(v), 0);
        enc.set_buffer(3, Some(bias), 0);
        enc.set_buffer(4, Some(out), 0);
        enc.set_bytes(5, 100, p.as_ptr() as *const c_void);
        let (threads, tg) = Self::attn_grid(p, kind, x);
        enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
    }



    pub fn gemv_sg_pipeline(&self, csplit: u32) -> ComputePipelineState {
        let key = (100 + csplit, 0);
        if let Some(p) = self.gemv_variants.borrow().get(&key) {
            return p.clone();
        }
        let fcv = metal::FunctionConstantValues::new();
        let four = 4u32;
        fcv.set_constant_value_at_index(
            &four as *const u32 as *const c_void,
            metal::MTLDataType::UInt,
            0,
        );
        fcv.set_constant_value_at_index(
            &csplit as *const u32 as *const c_void,
            metal::MTLDataType::UInt,
            1,
        );
        let f = self.lib.get_function("gemv_sg_t", Some(fcv)).expect("gemv_sg_t");
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .expect("gemv_sg_t pipeline");
        self.gemv_variants.borrow_mut().insert(key, p.clone());
        p
    }

    pub(crate) fn encode_gemv_sg(
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
        csplit: u32,
        tg: u64,
    ) {
        let enc = self.enc(cb, "gemm_sg");
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
        let cols_per_tg = (tg / 32) * csplit as u64 * 4;
        let tgs = (n as u64).div_ceil(cols_per_tg);
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
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

    pub(crate) fn encode_gemv_rows(
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
        let enc = self.enc(cb, "gemv_rows");
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

    pub(crate) fn encode_gemv_t(
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
        let enc = self.enc(cb, "gemv_t");
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


    pub fn search_cached(&self, comp: &crate::kernelset::Composition) -> (usize, crate::kernelset::Variant) {
        let key = comp.key();
        if let Some(c) = self.ksearch.borrow().get(&key) {
            return *c;
        }
        let r = crate::kernelset::search(self, &self.kernels, &crate::kernelset::IsolatedBench, comp)
            .unwrap_or((0, crate::kernelset::Variant([0; 5])));
        self.ksearch.borrow_mut().insert(key, r);
        r
    }

    pub fn kernel_index(&self, name: &str) -> usize {
        self.kernels.iter().position(|k| k.name() == name).expect("kernel name")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn contract_into(&self, cb: &metal::CommandBufferRef, a: &Buffer, b: &Buffer, c: &Buffer, m: usize, k: usize, n: usize, ap: [i32; 4], bp: [i32; 4], cp: [i32; 4], batch: &[i32], nbatch: i64) {
        let shape = crate::native_exec::ContractShape { m, n, ap, bp, cp, batch: batch.to_vec(), nbatch };
        let comp = crate::kernelset::Composition::Contraction { shape, kext: k as i64 };
        let (kidx, variant) = self.search_cached(&comp);
        self.run_kernel(cb, kidx, variant, &[a.clone(), b.clone(), c.clone()], &comp);
    }

    pub fn kernel_name(&self, kidx: usize) -> &'static str {
        self.kernels[kidx].name()
    }

    pub fn kernel_breaks_cb(&self, kidx: usize) -> bool {
        self.kernels[kidx].breaks_cb()
    }

    pub fn run_kernel(&self, cb: &metal::CommandBufferRef, kidx: usize, v: crate::kernelset::Variant, bufs: &[Buffer], comp: &crate::kernelset::Composition) {
        self.kernels[kidx].dispatch(self, cb, bufs, v, comp);
    }




    pub fn begin_sampling(&self, cap: u64) {
        let ts = self
            .device
            .counter_sets()
            .into_iter()
            .find(|c| c.name() == "timestamp")
            .expect("timestamp counter set");
        let desc = metal::CounterSampleBufferDescriptor::new();
        desc.set_counter_set(&ts);
        desc.set_sample_count(cap * 2);
        desc.set_storage_mode(metal::MTLStorageMode::Shared);
        let buf = self
            .device
            .new_counter_sample_buffer_with_descriptor(&desc)
            .expect("sample buffer");
        *self.sample_ctx.borrow_mut() = Some(SampleCtx { buf, labels: Vec::new() });
    }

    pub fn end_sampling(&self) -> Option<SampleCtx> {
        self.sample_ctx.borrow_mut().take()
    }

    pub fn enc<'a>(
        &self,
        cb: &'a metal::CommandBufferRef,
        label: &str,
    ) -> &'a metal::ComputeCommandEncoderRef {
        let mut ctx = self.sample_ctx.borrow_mut();
        if let Some(c) = ctx.as_mut() {
            let i = c.labels.len() as u64;
            if (i + 1) * 2 <= c.buf.sample_count() {
                c.labels.push(label.to_string());
                let desc = metal::ComputePassDescriptor::new();
                let att = desc.sample_buffer_attachments().object_at(0).unwrap();
                att.set_sample_buffer(&c.buf);
                att.set_start_of_encoder_sample_index(i * 2);
                att.set_end_of_encoder_sample_index(i * 2 + 1);
                return cb.compute_command_encoder_with_descriptor(&desc);
            }
        }
        cb.new_compute_command_encoder()
    }

    pub fn sample_timestamps(&self) -> (u64, u64) {
        let (mut c, mut g) = (0u64, 0u64);
        self.device.sample_timestamps(&mut c, &mut g);
        (c, g)
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

    pub fn ensure_reduce_fused(
        &mut self,
        name: &'static str,
        n_ext: usize,
        stages: &[(String, bool, String)],
        out_expr: &str,
    ) {
        if self.pipelines.contains_key(name) {
            return;
        }
        let b = 1 + n_ext;
        let mut sig = String::from("    device float* OUT [[buffer(0)]],\n");
        for k in 0..n_ext {
            sig += &format!("    device const float* in{k} [[buffer({})]],\n", 1 + k);
        }
        sig += &format!("    constant uint& rows [[buffer({b})]],\n");
        sig += &format!("    constant uint& cols [[buffer({})]],\n", b + 1);
        for k in 0..n_ext {
            sig += &format!("    constant int* m{k} [[buffer({})]],\n", b + 2 + k);
        }
        let mut body = String::new();
        for (k, (pre, rmax, vexpr)) in stages.iter().enumerate() {
            let (init, acc_c, tree_c) = if *rmax {
                ("-INFINITY", "acc = fmax(acc, pre);", "sh[lane] = fmax(sh[lane], sh[lane+s]);")
            } else {
                ("0.0f", "acc = acc + pre;", "sh[lane] = sh[lane] + sh[lane+s];")
            };
            body += &format!(
                "    float t{k};\n    {{ float acc = {init};\n      for (uint j = lane; j < cols; j += 256) {{ uint e = row*cols + j; float pre = {pre}; {acc_c} }}\n      sh[lane] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);\n      for (uint s = 128; s > 0; s >>= 1) {{ if (lane < s) {{ {tree_c} }} threadgroup_barrier(mem_flags::mem_threadgroup); }}\n      t{k} = sh[0]; threadgroup_barrier(mem_flags::mem_threadgroup); }}\n    float v{k} = {vexpr};\n"
            );
        }
        let src = format!(
            "{PREAMBLE}\nkernel void {name}(\n{sig}    uint lane [[thread_position_in_threadgroup]],\n    uint tgid [[threadgroup_position_in_grid]])\n{{\n    uint row = tgid; if (row >= rows) return;\n    threadgroup float sh[256];\n{body}    for (uint j = lane; j < cols; j += 256) {{ uint e = row*cols + j; OUT[e] = {out_expr}; }}\n}}\n"
        );
        let pipe = self.external_pipeline(&src, false, name);
        self.pipelines.insert(name, pipe);
    }

    pub fn ensure_fused(&mut self, name: &'static str, hdr: &str, body: &str, root: usize) {
        if self.pipelines.contains_key(name) {
            return;
        }
        let src = format!(
            "{PREAMBLE}\nkernel void {name}(\n{hdr}    uint gid [[thread_position_in_grid]])\n{{\n    if (gid >= n) return;\n{body}    OUT[gid] = v{root};\n}}\n"
        );
        let pipe = self.external_pipeline(&src, false, name);
        self.pipelines.insert(name, pipe);
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
        let enc = self.enc(cb, "matmul");
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

    #[allow(clippy::too_many_arguments)]

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bmm_scalar(
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
        batch: &[i32],
        nbatch: u64,
    ) {
        let enc = self.enc(cb, "bmm");
        enc.set_compute_pipeline_state(self.pipe("bmm"));
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
        enc.set_bytes(9, (batch.len() * 4) as u64, batch.as_ptr() as *const c_void);
        let tg = MTLSize::new(TILE, TILE, 1);
        let grid = MTLSize::new((n as u64).div_ceil(TILE), (m as u64).div_ceil(TILE), nbatch);
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
        let enc = self.enc(cb, "ew");
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
        let enc = self.enc(cb, "ew");
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
        let enc = self.enc(cb, "ew");
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
        let enc = self.enc(cb, "ew");
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
        let enc = self.enc(cb, "ew");
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
