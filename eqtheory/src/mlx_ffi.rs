use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

unsafe extern "C" {
    fn mlxllama_new(vocab: c_int, d: c_int, ff: c_int, h: c_int, kv: c_int, layers: c_int, eps: f32, scale: f32) -> *mut c_void;
    fn mlxllama_set_weight(h: *mut c_void, name: *const c_char, data: *const f32, shape: *const i64, ndim: c_int);
    fn mlxllama_prefill(h: *mut c_void, ids: *const c_int, n: c_int, out_last: *mut f32);
    fn mlxllama_decode(h: *mut c_void, tok: c_int, out: *mut f32);
    fn mlxllama_free(h: *mut c_void);
    fn mlxop_matmul(a: *const f32, b: *const f32, out: *mut f32, m: c_int, k: c_int, n: c_int);
    fn mlxop_sdpa(q: *const f32, k: *const f32, v: *const f32, mask: *const f32, out: *mut f32, p: *const c_int);
}

pub fn mlx_matmul(a: *const f32, b: *const f32, out: *mut f32, m: usize, k: usize, n: usize) {
    unsafe { mlxop_matmul(a, b, out, m as c_int, k as c_int, n as c_int) };
}

pub fn mlx_sdpa(q: *const f32, k: *const f32, v: *const f32, mask: *const f32, out: *mut f32, p: &[i32; 25]) {
    unsafe { mlxop_sdpa(q, k, v, mask, out, p.as_ptr()) };
}

pub struct MlxLlama {
    h: *mut c_void,
    pub vocab: usize,
}

impl MlxLlama {
    pub fn new(vocab: i32, d: i32, ff: i32, heads: i32, kv: i32, layers: i32, eps: f32, scale: f32) -> Self {
        let h = unsafe { mlxllama_new(vocab, d, ff, heads, kv, layers, eps, scale) };
        MlxLlama { h, vocab: vocab as usize }
    }

    pub fn set_weight(&mut self, name: &str, data: &[f32], shape: &[i64]) {
        let cn = CString::new(name).unwrap();
        unsafe {
            mlxllama_set_weight(self.h, cn.as_ptr(), data.as_ptr(), shape.as_ptr(), shape.len() as c_int);
        }
    }

    pub fn prefill(&mut self, ids: &[i32]) -> Vec<f32> {
        let mut out = vec![0f32; self.vocab];
        unsafe { mlxllama_prefill(self.h, ids.as_ptr(), ids.len() as c_int, out.as_mut_ptr()) };
        out
    }

    pub fn decode(&mut self, tok: i32) -> Vec<f32> {
        let mut out = vec![0f32; self.vocab];
        unsafe { mlxllama_decode(self.h, tok, out.as_mut_ptr()) };
        out
    }
}

impl Drop for MlxLlama {
    fn drop(&mut self) {
        unsafe { mlxllama_free(self.h) };
    }
}
