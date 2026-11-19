import json
import os
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FAT = {
    "vocab": 32000,
    "d": 1024,
    "ff": 2752,
    "n_heads": 16,
    "n_kv_heads": 4,
    "layers": 8,
    "eps": 1e-5,
}


def load_weights(src_dir):
    weights = {}
    for line in open(os.path.join(src_dir, "manifest.txt")):
        f = line.strip().split("|")
        if f[0] != "input" or f[1].startswith("arg."):
            continue
        shape = tuple(int(x) for x in f[3].split(","))
        raw = np.fromfile(os.path.join(src_dir, f[4]), dtype=np.float32)
        weights[f[1]] = raw.reshape(shape)
    return weights


class Graph:
    def __init__(self):
        self.nodes = []
        self.inits = []
        self.n = 0

    def init(self, name, arr):
        self.inits.append(numpy_helper.from_array(arr.astype(np.float32), name))
        return name

    def const_i64(self, name, arr):
        self.inits.append(numpy_helper.from_array(np.array(arr, dtype=np.int64), name))
        return name

    def node(self, op, ins, out=None, **attrs):
        out = out or f"t{self.n}"
        self.n += 1
        self.nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    def matmul(self, a, b):
        return self.node("MatMul", [a, b])

    def add(self, a, b):
        return self.node("Add", [a, b])

    def mul(self, a, b):
        return self.node("Mul", [a, b])

    def reshape(self, a, shape_name):
        return self.node("Reshape", [a, shape_name])

    def transpose(self, a, perm):
        return self.node("Transpose", [a], perm=perm)


def rmsnorm(g, x, eps_name, d, inv_d_name):
    sq = g.mul(x, x)
    ms = g.node("ReduceMean", [sq], axes=[-1], keepdims=1)
    mse = g.add(ms, eps_name)
    r = g.node("Sqrt", [mse])
    return g.node("Div", [x, r])


def build_onnx(cfg, weights, out_path):
    d = cfg["d"]
    ff = cfg["ff"]
    nkv = cfg["n_kv_heads"]
    nh = cfg["n_heads"]
    groups = nh // nkv
    hd = d // nh
    kvd = nkv * hd
    vocab = cfg["vocab"]
    L = cfg["layers"]
    scale = 1.0 / np.sqrt(hd)

    g = Graph()
    eps_name = g.init("eps", np.array([cfg["eps"]], dtype=np.float32))
    scale_name = g.init("scale", np.array([scale], dtype=np.float32))
    q_shape = g.const_i64("q_shape", [-1, nkv, groups, hd])
    kv_shape = g.const_i64("kv_shape", [-1, nkv, hd])
    k_bc = g.const_i64("k_bc", [nkv, 1, hd, -1])
    v_bc = g.const_i64("v_bc", [nkv, 1, -1, hd])
    o_shape = g.const_i64("o_shape", [-1, d])
    mask_axes = g.const_i64("mask_axes", [0, 1])

    embed = g.init("model.embed_tokens.weight", weights["model.embed_tokens.weight"])
    embed_t = g.transpose(embed, [1, 0])

    ids = "input_ids"
    x = g.node("Gather", [embed, ids], axis=0)

    present = []
    for l in range(L):
        p = f"model.layers.{l}"
        wq = g.init(f"{p}.self_attn.q_proj.weight", weights[f"{p}.self_attn.q_proj.weight"])
        wk = g.init(f"{p}.self_attn.k_proj.weight", weights[f"{p}.self_attn.k_proj.weight"])
        wv = g.init(f"{p}.self_attn.v_proj.weight", weights[f"{p}.self_attn.v_proj.weight"])
        wo = g.init(f"{p}.self_attn.o_proj.weight", weights[f"{p}.self_attn.o_proj.weight"])
        wg = g.init(f"{p}.mlp.gate_proj.weight", weights[f"{p}.mlp.gate_proj.weight"])
        wu = g.init(f"{p}.mlp.up_proj.weight", weights[f"{p}.mlp.up_proj.weight"])
        wd = g.init(f"{p}.mlp.down_proj.weight", weights[f"{p}.mlp.down_proj.weight"])

        normed = rmsnorm(g, x, eps_name, d, None)
        q = g.matmul(normed, wq)
        k_new = g.matmul(normed, wk)
        v_new = g.matmul(normed, wv)
        k_all = g.node("Concat", [f"past_k_{l}", k_new], axis=0)
        v_all = g.node("Concat", [f"past_v_{l}", v_new], axis=0)
        present.append((k_all, v_all))

        qh = g.transpose(g.reshape(q, q_shape), [1, 2, 0, 3])
        kh = g.reshape(g.transpose(g.reshape(k_all, kv_shape), [1, 2, 0]), k_bc)
        scores = g.mul(g.matmul(qh, kh), scale_name)
        mask4 = g.node("Unsqueeze", ["attn_mask", mask_axes])
        scores = g.add(scores, mask4)
        w = g.node("Softmax", [scores], axis=3)
        vh = g.reshape(g.transpose(g.reshape(v_all, kv_shape), [1, 0, 2]), v_bc)
        o = g.transpose(g.matmul(w, vh), [2, 0, 1, 3])
        o = g.reshape(o, o_shape)
        x = g.add(x, g.matmul(o, wo))

        ff_in = rmsnorm(g, x, eps_name, d, None)
        gate = g.matmul(ff_in, wg)
        silu = g.mul(gate, g.node("Sigmoid", [gate]))
        up = g.matmul(ff_in, wu)
        h = g.mul(silu, up)
        x = g.add(x, g.matmul(h, wd))

    normed = rmsnorm(g, x, eps_name, d, None)
    logits = g.node("MatMul", [normed, embed_t], out="logits")

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["seq"]),
        helper.make_tensor_value_info("attn_mask", TensorProto.FLOAT, ["seq", "total"]),
    ]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["seq", vocab])]
    for l in range(L):
        inputs.append(helper.make_tensor_value_info(f"past_k_{l}", TensorProto.FLOAT, ["past", kvd]))
        inputs.append(helper.make_tensor_value_info(f"past_v_{l}", TensorProto.FLOAT, ["past", kvd]))
        outputs.append(helper.make_tensor_value_info(present[l][0], TensorProto.FLOAT, ["total", kvd]))
        outputs.append(helper.make_tensor_value_info(present[l][1], TensorProto.FLOAT, ["total", kvd]))

    graph = helper.make_graph(g.nodes, "fat", inputs, outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.save(model, out_path)
    cfg_out = dict(cfg)
    cfg_out["present_names"] = [[present[l][0], present[l][1]] for l in range(L)]
    json.dump(cfg_out, open(out_path.replace(".onnx", ".json"), "w"), indent=2)
    return model


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.environ["FATMODEL"]
    out = sys.argv[2] if len(sys.argv) > 2 else "models/fat/model.onnx"
    w = load_weights(src)
    build_onnx(FAT, w, out)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
