import os
import time
import numpy as np
import onnx
from onnx import numpy_helper
import mlx.core as mx

VOCAB, D, FF, H, KV, LAYERS = 32000, 1024, 2752, 16, 4, 8
HD = D // H
EPS = 1e-5
SCALE = 1.0 / np.sqrt(HD)


def load_weights(onnx_path):
    m = onnx.load(onnx_path)
    w = {}
    for init in m.graph.initializer:
        if init.data_type == 1 and int(np.prod(init.dims)) > 1:
            w[init.name] = mx.array(numpy_helper.to_array(init))
    return w


def run_mlx(onnx_path, p, g, c, out_dir):
    w = load_weights(onnx_path)
    embed = w["model.embed_tokens.weight"]
    ones = mx.ones((D,))

    def wl(l, n):
        return w[f"model.layers.{l}.{n}.weight"]

    def rms(x):
        return mx.fast.rms_norm(x, ones, EPS)

    def layer(l, x, kc, vc, mask):
        nm = rms(x)
        q = nm @ wl(l, "self_attn.q_proj")
        k = nm @ wl(l, "self_attn.k_proj")
        v = nm @ wl(l, "self_attn.v_proj")
        kc = k if kc is None else mx.concatenate([kc, k], axis=0)
        vc = v if vc is None else mx.concatenate([vc, v], axis=0)
        s, t = x.shape[0], kc.shape[0]
        qh = q.reshape(s, H, HD).transpose(1, 0, 2)[None]
        kh = kc.reshape(t, KV, HD).transpose(1, 0, 2)[None]
        vh = vc.reshape(t, KV, HD).transpose(1, 0, 2)[None]
        o = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=SCALE, mask=mask)
        attn = o[0].transpose(1, 0, 2).reshape(s, H * HD)
        x = x + attn @ wl(l, "self_attn.o_proj")
        nm2 = rms(x)
        gate = nm2 @ wl(l, "mlp.gate_proj")
        up = nm2 @ wl(l, "mlp.up_proj")
        h = (gate * mx.sigmoid(gate) * up) @ wl(l, "mlp.down_proj")
        return x + h, kc, vc

    def prefill(ids, caches):
        x = embed[ids]
        new = []
        for l in range(LAYERS):
            x, kc, vc = layer(l, x, caches[l][0], caches[l][1], "causal")
            new.append((kc, vc))
        return rms(x) @ embed.T, new

    def decode(tok, caches):
        x = embed[tok]
        new = []
        for l in range(LAYERS):
            x, kc, vc = layer(l, x, caches[l][0], caches[l][1], None)
            new.append((kc, vc))
        return rms(x) @ embed.T, new

    prompt = mx.array([(i * 7 + 3) % VOCAB for i in range(p)], dtype=mx.int32)
    gen = [(1000 + 13 * t) % VOCAB for t in range(g)]

    def session():
        caches = [(None, None)] * LAYERS
        logits_p, caches = prefill(prompt, caches)
        mx.eval(logits_p, *[x for pair in caches for x in pair])
        last = np.array(logits_p[p - 1])
        first = final = None
        for t, tok in enumerate(gen):
            logits, caches = decode(mx.array([tok], dtype=mx.int32), caches)
            mx.eval(logits, *[x for pair in caches for x in pair])
            if t == 0:
                first = np.array(logits[0])
            if t == g - 1:
                final = np.array(logits[0])
        return last, first, final

    session()
    caches = [(None, None)] * LAYERS
    t0 = time.perf_counter()
    logits_p, caches = prefill(prompt, caches)
    mx.eval(logits_p, *[x for pair in caches for x in pair])
    ttft = (time.perf_counter() - t0) * 1e3
    last = np.array(logits_p[p - 1])
    first = final = None
    t1 = time.perf_counter()
    for t, tok in enumerate(gen):
        logits, caches = decode(mx.array([tok], dtype=mx.int32), caches)
        mx.eval(logits, *[x for pair in caches for x in pair])
        if t == 0:
            first = np.array(logits[0])
        if t == g - 1:
            final = np.array(logits[0])
    gen_ms = (time.perf_counter() - t1) * 1e3

    logits = np.stack([last, first, final]).astype(np.float32)
    logits.tofile(os.path.join(out_dir, "mlx_logits.bin"))
    return {
        "name": "mlx",
        "platform": "metal",
        "ttft_ms": ttft,
        "decode_tok_s": g / (gen_ms / 1e3),
        "e2e_ms": ttft + gen_ms,
        "logits_file": "mlx_logits.bin",
    }


if __name__ == "__main__":
    import sys, json
    onnx_path, p, g, c, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    print(json.dumps(run_mlx(onnx_path, p, g, c, out)))
