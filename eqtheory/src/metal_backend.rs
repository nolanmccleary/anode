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
    gemv_choice: std::cell::RefCell<HashMap<(usize, usize, bool), (u32, u32, u32, u64)>>,
    rows_variants: std::cell::RefCell<HashMap<u32, ComputePipelineState>>,
    attn_variants: std::cell::RefCell<HashMap<(u32, u32), ComputePipelineState>>,
    attn_choice: std::cell::RefCell<HashMap<(usize, usize, usize, bool), (u32, u32)>>,
    gemm_variants: std::cell::RefCell<HashMap<(u32, u32, u32, u32), ComputePipelineState>>,
    gemm_choice: std::cell::RefCell<HashMap<(usize, usize, usize), Option<(u32, u32, u32, u32)>>>,
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
            gemv_choice: Default::default(),
            rows_variants: Default::default(),
            attn_variants: Default::default(),
            attn_choice: Default::default(),
            gemm_variants: Default::default(),
            gemm_choice: Default::default(),
        };
        let _ = me.gemv_pipeline(4, 8);
        let _ = me.gemv_rows_pipeline(4);
        me
    }

    pub fn gemm_pipeline(&self, v: (u32, u32, u32, u32)) -> ComputePipelineState {
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
        let f = self.lib.get_function("gemm_sg", Some(fcv)).expect("gemm_sg fc");
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .expect("gemm_sg pipeline");
        self.gemm_variants.borrow_mut().insert(v, p.clone());
        p
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gemm_sg(
        &self,
        cb: &metal::CommandBufferRef,
        v: (u32, u32, u32, u32),
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
        let pipe = self.gemm_pipeline(v);
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(b), 0);
        enc.set_buffer(2, Some(c), 0);
        let dims = [m as i32, k as i32, n as i32, 0];
        enc.set_bytes(3, 16, dims.as_ptr() as *const c_void);
        enc.set_bytes(4, 16, ap.as_ptr() as *const c_void);
        enc.set_bytes(5, 16, bp.as_ptr() as *const c_void);
        enc.set_bytes(6, 16, cp.as_ptr() as *const c_void);
        let bm = (v.0 * v.2 * 8) as u64;
        let bn = (v.1 * v.3 * 8) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(n as u64 / bn, m as u64 / bm, 1),
            MTLSize::new((32 * v.0 * v.1) as u64, 1, 1),
        );
        enc.end_encoding();
    }

    pub fn tune_gemm(&self, m: usize, k: usize, n: usize, ap: [i32; 4], bp: [i32; 4], cp: [i32; 4]) {
        let key = (m, k, n);
        if self.gemm_choice.borrow().contains_key(&key) {
            return;
        }
        let ap0 = [0, ap[1], ap[2], 0];
        let bp0 = [0, bp[1], bp[2], 0];
        let cp0 = [0, cp[1], cp[2], 0];
        let plan = crate::atoms::mma_plan(&crate::atoms::SG_F32_8X8X8, m, k, n, ap0, bp0, cp0);
        let candidates = crate::atoms::mma_candidates(m, n);
        if plan.is_none() || candidates.is_empty() || std::env::var("GEMM_NOTUNE").is_ok() {
            self.gemm_choice.borrow_mut().insert(key, None);
            return;
        }
        let foot = |s1: i32, n1: usize, s2: i32, n2: usize| -> usize {
            1 + s1.unsigned_abs() as usize * (n1 - 1) + s2.unsigned_abs() as usize * (n2 - 1)
        };
        let afoot = (foot(ap[1], m, ap[2], k) + 3) & !3;
        let bfoot = (foot(bp[1], k, bp[2], n) + 3) & !3;
        let cfoot = foot(cp[1], m, 1, n);
        let copies = (96_000_000usize / ((afoot + bfoot) * 4)).clamp(1, 8);
        let ascratch = self.alloc_bytes(afoot * 4 * copies);
        let bscratch = self.alloc_bytes(bfoot * 4 * copies);
        let cscratch = self.alloc_bytes(cfoot * 4);
        unsafe {
            std::ptr::write_bytes(ascratch.contents() as *mut u8, 0, afoot * 4 * copies);
            std::ptr::write_bytes(bscratch.contents() as *mut u8, 0, bfoot * 4 * copies);
        }
        let mut best: (f64, Option<(u32, u32, u32, u32)>) = (f64::INFINITY, None);
        let mut race = |v: Option<(u32, u32, u32, u32)>, best: &mut (f64, Option<(u32, u32, u32, u32)>)| {
            let run = |iters: usize, offs: usize| {
                let cb = self.queue.new_command_buffer();
                for it in 0..iters {
                    let c = (it + offs) % copies;
                    let mut app = ap0;
                    let mut bpp = bp0;
                    app[0] = (c * afoot) as i32;
                    bpp[0] = (c * bfoot) as i32;
                    match v {
                        Some(vv) => self.encode_gemm_sg(cb, vv, &ascratch, &bscratch, &cscratch, m, k, n, app, bpp, cp0),
                        None => self.matmul_into(cb, &ascratch, &bscratch, &cscratch, m, k, n, app, bpp, cp0),
                    }
                }
                cb.commit();
                cb.wait_until_completed();
            };
            run(1, 0);
            let t = std::time::Instant::now();
            run(2, 1);
            let ms = t.elapsed().as_secs_f64() * 1e3 / 2.0;
            if ms < best.0 {
                *best = (ms, v);
            }
            ms
        };
        let scalar_ms = race(None, &mut best);
        for v in candidates {
            race(Some(v), &mut best);
        }
        if std::env::var("GEMMTUNE").is_ok() {
            let gf = 2.0 * m as f64 * k as f64 * n as f64 / (best.0 / 1e3) / 1e9;
            eprintln!(
                "gemm tune M={m} K={k} N={n}: {:?} ({:.3} ms, {:.0} GF/s; scalar {:.3} ms)",
                best.1, best.0, gf, scalar_ms
            );
        }
        self.gemm_choice.borrow_mut().insert(key, best.1);
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

    fn encode_attn(
        &self,
        cb: &metal::CommandBufferRef,
        pipe: &ComputePipelineState,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        bias: &Buffer,
        out: &Buffer,
        p: &[i32; 25],
        split: u32,
    ) {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(k), 0);
        enc.set_buffer(2, Some(v), 0);
        enc.set_buffer(3, Some(bias), 0);
        enc.set_buffer(4, Some(out), 0);
        enc.set_bytes(5, 100, p.as_ptr() as *const c_void);
        let jobs = (p[0] * p[1] * p[2]) as u64;
        let threads = jobs.div_ceil(8 / split as u64) * 256;
        enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }

    pub fn tune_attn(&self, p: &[i32; 25], vec_ok: bool) -> (u32, u32) {
        let (kv, g, s, ctx, hd) = (
            p[0] as usize,
            p[1] as usize,
            p[2] as usize,
            p[3] as usize,
            p[4] as usize,
        );
        let jobs = kv * g * s;
        let key = (jobs, ctx, hd, vec_ok);
        if let Some(c) = self.attn_choice.borrow().get(&key) {
            return *c;
        }
        let vec = if vec_ok { 4u32 } else { 1u32 };
        if std::env::var("ATTN_NOTUNE").is_ok() {
            self.attn_choice.borrow_mut().insert(key, (1, vec));
            return (1, vec);
        }
        let foot = |s0: i32, n0: usize, s1: i32, n1: usize, s2: i32, n2: usize, s3: i32, n3: usize| -> usize {
            1 + s0.max(0) as usize * (n0 - 1)
                + s1.max(0) as usize * (n1 - 1)
                + s2.max(0) as usize * (n2 - 1)
                + s3.max(0) as usize * (n3 - 1)
        };
        let kfoot = (foot(p[11], kv, p[12], ctx, p[13], hd, 0, 1) + 3) & !3;
        let vfoot = (foot(p[15], kv, p[16], ctx, p[17], hd, 0, 1) + 3) & !3;
        let qfoot = foot(p[6], kv, p[7], g, p[8], s, p[9], hd);
        let bfoot = foot(p[19], kv, p[20], g, p[21], s, p[22], ctx);
        let copies = (96_000_000usize / ((kfoot + vfoot) * 4)).clamp(1, 8);
        let kscratch = self.alloc_bytes(kfoot * 4 * copies);
        let vscratch = self.alloc_bytes(vfoot * 4 * copies);
        let qscratch = self.alloc_bytes(qfoot * 4);
        let bscratch = self.alloc_bytes(bfoot * 4);
        let oscratch = self.alloc_bytes(jobs * hd * 4);
        unsafe {
            std::ptr::write_bytes(qscratch.contents() as *mut u8, 0, qfoot * 4);
            std::ptr::write_bytes(bscratch.contents() as *mut u8, 0, bfoot * 4);
            std::ptr::write_bytes(kscratch.contents() as *mut u8, 0, kfoot * 4 * copies);
            std::ptr::write_bytes(vscratch.contents() as *mut u8, 0, vfoot * 4 * copies);
        }
        let mut best = (f64::INFINITY, (1u32, vec));
        for split in [1u32, 2, 4, 8] {
            if split as usize > ctx {
                continue;
            }
            let pipe = self.attn_pipeline(split, vec);
            let run = |iters: usize, offs: usize| {
                let cb = self.queue.new_command_buffer();
                for it in 0..iters {
                    let c = (it + offs) % copies;
                    let mut pp = *p;
                    pp[5] = 0;
                    pp[10] = (c * kfoot) as i32;
                    pp[14] = (c * vfoot) as i32;
                    pp[18] = 0;
                    self.encode_attn(cb, &pipe, &qscratch, &kscratch, &vscratch, &bscratch, &oscratch, &pp, split);
                }
                cb.commit();
                cb.wait_until_completed();
            };
            run(2, 0);
            let t = std::time::Instant::now();
            run(4, 2);
            let ms = t.elapsed().as_secs_f64() * 1e3 / 4.0;
            if ms < best.0 {
                best = (ms, (split, vec));
            }
        }
        if std::env::var("ATTNTUNE").is_ok() {
            eprintln!(
                "attn tune jobs={jobs} ctx={ctx} hd={hd} vec={vec}: split={} ({:.3} ms)",
                best.1.0, best.0
            );
        }
        self.attn_choice.borrow_mut().insert(key, best.1);
        best.1
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

    fn encode_gemv_sg(
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
        let stride_f = (k * n + 3) & !3;
        let bytes = stride_f * 4;
        let copies = (96_000_000usize / bytes).clamp(1, 8);
        let wscratch = self.alloc_bytes(bytes * copies);
        let out_scratch = self.alloc_bytes(n * 4);
        let xap = [0, ap[2], 0, 0];
        let mut best = (f64::INFINITY, (0u32, 4u32, 8u32, 256u64));
        let col_vec4 = bp[0] % 4 == 0 && bp[1] % 4 == 0;
        let rows_vec4 = bp[0] % 4 == 0 && bp[2] % 4 == 0 && k % 4 == 0 && ap[2] == 1;
        let candidates: Vec<(u32, u32, u32, u64)> = if rows {
            let rv = if rows_vec4 { 4u32 } else { 1u32 };
            let mut v = Vec::new();
            for tg in [128u64, 256u64] {
                v.push((2u32, rv, 1u32, tg));
            }
            v
        } else {
            let vecs: &[u32] = if col_vec4 { &[4, 8] } else { &[4] };
            let mut v = Vec::new();
            for tg in [128u64, 256u64] {
                for ncolg in [4u32, 8u32, 16u32] {
                    let ks = tg / ncolg as u64;
                    if !(8..=64).contains(&ks) {
                        continue;
                    }
                    for vec in vecs {
                        v.push((0u32, *vec, ncolg, tg));
                    }
                }
                for csplit in [4u32, 8u32, 16u32] {
                    v.push((1u32, 4u32, csplit, tg));
                }
            }
            v
        };
        for (kind, vec, ncolg, tg) in candidates {
            let run = |iters: usize, offs: usize| {
                let cb = self.queue.new_command_buffer();
                for it in 0..iters {
                    let off = (((it + offs) % copies) * stride_f) as i32;
                    let enc_w = &wscratch;
                    let wbp0 = if rows {
                        [off, 1, bp[2], 0]
                    } else {
                        [off, bp[1], 1, 0]
                    };
                    match kind {
                        2 => {
                            let pipe = self.gemv_rows_pipeline(vec);
                            self.encode_gemv_rows(cb, &pipe, a, enc_w, &out_scratch, k, n, xap, wbp0, 0, tg);
                        }
                        1 => {
                            let pipe = self.gemv_sg_pipeline(ncolg);
                            self.encode_gemv_sg(cb, &pipe, a, enc_w, &out_scratch, k, n, xap, wbp0, 0, ncolg, tg);
                        }
                        _ => {
                            let pipe = self.gemv_pipeline(vec, ncolg);
                            self.encode_gemv_t(cb, &pipe, a, enc_w, &out_scratch, k, n, xap, wbp0, 0, vec, ncolg, tg);
                        }
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
                best = (ms, (kind, vec, ncolg, tg));
            }
        }
        if std::env::var("GEMVTUNE").is_ok() {
            eprintln!(
                "gemv tune K={k} N={n} rows={rows}: kind={} vec={} p={} tg={} ({:.3} ms, {:.1} GB/s)",
                best.1.0,
                best.1.1,
                best.1.2,
                best.1.3,
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
        if std::env::var("GENERAL_ONLY").is_ok() {
            self.matmul_into(cb, a, b, c, m, k, n, ap, bp, cp);
            return;
        }
        if m == 1 && bp[2] == 1 && cp[2] == 1 {
            let choice = self.gemv_choice.borrow().get(&(k, n, true)).copied();
            let (kind, vec, p2, tg) = choice.unwrap_or((0, 4, 8, 256));
            let xap = [ap[0], ap[2], 0, 0];
            let wbp = [bp[0], bp[1], 0, 0];
            if kind == 1 {
                let pipe = self.gemv_sg_pipeline(p2);
                self.encode_gemv_sg(cb, &pipe, a, b, c, k, n, xap, wbp, cp[0], p2, tg);
            } else {
                let pipe = self.gemv_pipeline(vec, p2);
                self.encode_gemv_t(cb, &pipe, a, b, c, k, n, xap, wbp, cp[0], vec, p2, tg);
            }
        } else if m == 1 && bp[1] == 1 && cp[2] == 1 {
            let choice = self.gemv_choice.borrow().get(&(k, n, false)).copied();
            let (_, vec, _, tg) = choice.unwrap_or((2, 4, 1, 256));
            let pipe = self.gemv_rows_pipeline(vec);
            let xap = [ap[0], ap[2], 0, 0];
            let wbp = [bp[0], 1, bp[2], 0];
            self.encode_gemv_rows(cb, &pipe, a, b, c, k, n, xap, wbp, cp[0], tg);
        } else {
            let choice = self.gemm_choice.borrow().get(&(m, k, n)).copied().flatten();
            match choice {
                Some(v) => self.encode_gemm_sg(cb, v, a, b, c, m, k, n, ap, bp, cp),
                None => self.matmul_into(cb, a, b, c, m, k, n, ap, bp, cp),
            }
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

    #[allow(clippy::too_many_arguments)]
    pub fn bmm_into(
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
        let enc = cb.new_compute_command_encoder();
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
