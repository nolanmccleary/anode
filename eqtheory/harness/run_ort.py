import os
import time
import numpy as np
import onnxruntime as ort

VOCAB = 32000
KVD, LAYERS = 256, 8


def run_ort(onnx_path, p, g, c, out_dir, provider="CPUExecutionProvider"):
    prompt = np.array([(i * 7 + 3) % VOCAB for i in range(p)], dtype=np.int64)
    gen = [(1000 + 13 * t) % VOCAB for t in range(g)]
    short = provider.replace("ExecutionProvider", "").lower()

    so = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, so, providers=[provider])
    names = [o.name for o in sess.get_outputs()]
    kv_names = [n for n in names if n != "logits"]

    def session_once():
        past = {f"past_k_{l}": np.zeros((0, KVD), np.float32) for l in range(LAYERS)}
        past.update({f"past_v_{l}": np.zeros((0, KVD), np.float32) for l in range(LAYERS)})

        def step(ids, mask):
            feeds = {"input_ids": ids, "attn_mask": mask}
            feeds.update(past)
            outs = sess.run(None, feeds)
            res = dict(zip(names, outs))
            for l in range(LAYERS):
                past[f"past_k_{l}"] = res[kv_names[2 * l]]
                past[f"past_v_{l}"] = res[kv_names[2 * l + 1]]
            return res["logits"]

        pre_mask = np.triu(np.full((p, p), -1e10, np.float32), k=1)
        t0 = time.perf_counter()
        pl = step(prompt, pre_mask)
        ttft = (time.perf_counter() - t0) * 1e3
        last = np.array(pl[p - 1])

        first = final = None
        t1 = time.perf_counter()
        for t, tok in enumerate(gen):
            total = p + t + 1
            lg = step(np.array([tok], dtype=np.int64), np.zeros((1, total), np.float32))
            if t == 0:
                first = np.array(lg[0])
            if t == g - 1:
                final = np.array(lg[0])
        gen_ms = (time.perf_counter() - t1) * 1e3
        return ttft, gen_ms, np.stack([last, first, final])

    session_once()
    ttft, gen_ms, logits = session_once()
    logits_file = f"ort_{short}_logits.bin"
    logits.astype(np.float32).tofile(os.path.join(out_dir, logits_file))
    return {
        "name": f"onnxruntime-{short}",
        "platform": short,
        "ttft_ms": ttft,
        "decode_tok_s": g / (gen_ms / 1e3),
        "e2e_ms": ttft + gen_ms,
        "logits_file": logits_file,
        "logits": logits,
    }


if __name__ == "__main__":
    import sys, json
    onnx, p, g, c, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    provider = sys.argv[6] if len(sys.argv) > 6 else "CPUExecutionProvider"
    r = run_ort(onnx, p, g, c, out, provider)
    del r["logits"]
    print(json.dumps(r))
