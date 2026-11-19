import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "target/release/run_burn")


def run_burn(onnx_path, p, g, c, out_dir):
    weights = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "export")
    proc = subprocess.run(
        [BIN, weights, str(p), str(g), str(c), out_dir],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    import sys
    onnx, p, g, c, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    print(json.dumps(run_burn(onnx, p, g, c, out)))
