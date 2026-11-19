import sys
import numpy as np
import onnxruntime as ort

VOCAB, P, G, C = 32000, 512, 64, 640
KVD, LAYERS = 256, 8

onnx_path = sys.argv[1]
native_bin = sys.argv[2]

prompt = np.array([(i * 7 + 3) % VOCAB for i in range(P)], dtype=np.int64)
gen = [(1000 + 13 * t) % VOCAB for t in range(G)]

so = ort.SessionOptions()
sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

past = {f"past_k_{l}": np.zeros((0, KVD), np.float32) for l in range(LAYERS)}
past.update({f"past_v_{l}": np.zeros((0, KVD), np.float32) for l in range(LAYERS)})


def step(ids, mask):
    feeds = {"input_ids": ids, "attn_mask": mask}
    feeds.update(past)
    outs = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    res = dict(zip(names, outs))
    for l in range(LAYERS):
        kname = [n for n in names if n != "logits"][2 * l]
        vname = [n for n in names if n != "logits"][2 * l + 1]
        past[f"past_k_{l}"] = res[kname]
        past[f"past_v_{l}"] = res[vname]
    return res["logits"]


pre_mask = np.triu(np.full((P, P), -1e10, np.float32), k=1)
pre_logits = step(prompt, pre_mask)
prefill_last = pre_logits[P - 1]

first_logits = final_logits = None
for t, tok in enumerate(gen):
    total = P + t + 1
    mask = np.zeros((1, total), np.float32)
    lg = step(np.array([tok], dtype=np.int64), mask)
    if t == 0:
        first_logits = lg[0]
    if t == G - 1:
        final_logits = lg[0]

ref = np.fromfile(native_bin, dtype=np.float32).reshape(3, VOCAB)
rows = [("prefill-last", prefill_last), ("decode-first", first_logits), ("decode-final", final_logits)]
ok_all = True
for i, (name, got) in enumerate(rows):
    d = np.abs(got - ref[i]).max()
    peak = np.abs(ref[i]).max()
    ok = d <= 1e-2 * peak
    ok_all &= ok
    print(f"{name}: max abs diff vs cute {d:.3e} (peak {peak:.2e}, rel {d/peak:.2e}) -> {'MATCH' if ok else 'MISMATCH'}")
print("onnx vs cute session:", "ALL MATCH" if ok_all else "MISMATCH")
