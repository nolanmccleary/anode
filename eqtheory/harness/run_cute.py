import ctypes
import json
import os
import subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "target/release/libcute.dylib")
DRIVER = os.path.join(ROOT, "target/release/cutedriver")


def run_cute(onnx_path, p, g, c, out_dir):
    lib = ctypes.CDLL(LIB)
    lib.cute_serve.restype = ctypes.c_char_p
    lib.cute_serve.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    logits_path = os.path.join(out_dir, "cute_logits.bin")
    sock = lib.cute_serve(onnx_path.encode(), p, g, c, logits_path.encode()).decode()
    metrics_json = subprocess.run([DRIVER, sock], capture_output=True, text=True, check=True).stdout
    m = json.loads(metrics_json)
    m["logits"] = np.fromfile(logits_path, dtype=np.float32).reshape(3, -1)
    m["platform"] = "metal"
    m["logits_file"] = "cute_logits.bin"
    return m


if __name__ == "__main__":
    import sys
    onnx, p, g, c, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    r = run_cute(onnx, p, g, c, out)
    del r["logits"]
    print(json.dumps(r))
