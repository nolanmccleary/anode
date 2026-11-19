use eqtheory::session_run::{run_session, Workload};
use std::ffi::{CStr, CString};
use std::io::{Read, Write};
use std::os::raw::c_char;
use std::os::unix::net::UnixListener;
use std::time::{SystemTime, UNIX_EPOCH};

const VOCAB: i32 = 32000;

fn cstr(p: *const c_char) -> String {
    unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string()
}

#[unsafe(no_mangle)]
pub extern "C" fn cute_serve(onnx: *const c_char, p: i32, g: i32, c: i32, logits_out: *const c_char) -> *mut c_char {
    let onnx = cstr(onnx);
    let logits_out = cstr(logits_out);
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let sock = format!("/tmp/cute-{}-{}.sock", std::process::id(), nanos);
    let _ = std::fs::remove_file(&sock);
    let listener = UnixListener::bind(&sock).unwrap();
    let ret = sock.clone();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut trig = [0u8; 1];
        let _ = stream.read_exact(&mut trig);
        let json = run(&onnx, p, g, c, &logits_out);
        let b = json.as_bytes();
        stream.write_all(&(b.len() as u32).to_le_bytes()).unwrap();
        stream.write_all(b).unwrap();
        let _ = std::fs::remove_file(&sock);
    });
    CString::new(ret).unwrap().into_raw()
}

fn run(onnx: &str, p: i32, g: i32, c: i32, logits_out: &str) -> String {
    let bytes = std::fs::read(onnx).unwrap();
    let w = Workload {
        prompt: (0..p).map(|i| (i * 7 + 3) % VOCAB).collect(),
        gen_tokens: (0..g).map(|t| (1000 + 13 * t) % VOCAB).collect(),
        ctx: c as i64,
    };
    let m = run_session(&bytes, &w);
    let logits: Vec<u8> = m.last_logits.iter().chain(&m.first_logits).chain(&m.final_logits).flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(logits_out, logits).unwrap();
    format!(
        "{{\"name\":\"cute\",\"ttft_ms\":{:.3},\"decode_tok_s\":{:.3},\"e2e_ms\":{:.3},\"prefill_dispatches\":{},\"decode_dispatches\":{}}}",
        m.ttft_ms, m.decode_tok_s, m.e2e_ms, m.prefill_dispatches, m.decode_dispatches
    )
}
