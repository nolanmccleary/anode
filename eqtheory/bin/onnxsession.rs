use eqtheory::session_run::{run_session, Workload};

const VOCAB: i64 = 32000;
const P: usize = 512;
const G: usize = 64;
const C: i64 = 640;

fn main() {
    let path = std::env::args().nth(1).expect("usage: onnxsession <model.onnx>");
    let bytes = std::fs::read(&path).unwrap();
    let w = Workload {
        prompt: (0..P as i32).map(|i| (i * 7 + 3) % VOCAB as i32).collect(),
        gen_tokens: (0..G as i32).map(|t| (1000 + 13 * t) % VOCAB as i32).collect(),
        ctx: C,
    };
    let m = run_session(&bytes, &w);
    println!(
        "prefill: {} dispatches, TTFT {:.1} ms ({:.1} tok/s prompt)",
        m.prefill_dispatches, m.ttft_ms, P as f64 / (m.ttft_ms / 1e3)
    );
    println!(
        "decode: {} dispatches, {G} tokens -> {:.0} tok/s",
        m.decode_dispatches, m.decode_tok_s
    );
    println!("end-to-end: {:.1} ms", m.e2e_ms);
    std::fs::write(
        format!("{path}.session.bin"),
        m.last_logits.iter().chain(&m.first_logits).chain(&m.final_logits)
            .flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    ).unwrap();
}
