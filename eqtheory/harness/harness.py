import csv
import datetime
import json
import os
import re
import subprocess
import threading
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv/bin/python")
BENCH = os.path.join(ROOT, "harness")

P, G, C = 512, 64, 640
REPEATS = 3
FAIRNESS_REL_TOL = 1e-2
SAMPLE_S = 0.1


def _gpu_util():
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True, timeout=2).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _swap_used_mb():
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=2).stdout
        m = re.search(r'used = ([\d.]+)M', out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _mem_free_pct():
    try:
        out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=2).stdout
        m = re.search(r'free percentage:\s*(\d+)%', out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _sample(stop, acc):
    while not stop.is_set():
        g, f, s = _gpu_util(), _mem_free_pct(), _swap_used_mb()
        if g is not None:
            acc["gpu"].append(g)
        if f is not None:
            acc["free"].append(f)
        if s is not None:
            acc["swap"].append(s)
        stop.wait(SAMPLE_S)


def _subrun(script, onnx, out_dir, extra=(), label=None):
    name = label or script
    runs = []
    gpu_pk, free_lo, swap_pk = [], [], []
    for _ in range(REPEATS):
        acc = {"gpu": [], "free": [], "swap": []}
        stop = threading.Event()
        t = threading.Thread(target=_sample, args=(stop, acc), daemon=True)
        proc = subprocess.Popen(
            [PY, os.path.join(BENCH, script), onnx, str(P), str(G), str(C), out_dir, *extra],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        t.start()
        out, err = proc.communicate()
        stop.set()
        t.join(timeout=1)
        if proc.returncode != 0:
            tail = err.strip().splitlines()[-1] if err.strip() else "failed"
            return {"name": name, "error": tail, "available": False}
        runs.append(json.loads(out.strip().splitlines()[-1]))
        if acc["gpu"]:
            gpu_pk.append(max(acc["gpu"]))
        if acc["free"]:
            free_lo.append(min(acc["free"]))
        if acc["swap"]:
            swap_pk.append(max(acc["swap"]))
    med = lambda k: float(np.median([r[k] for r in runs]))
    best_lo = lambda k: float(np.min([r[k] for r in runs]))
    best_hi = lambda k: float(np.max([r[k] for r in runs]))
    agg = lambda v: float(np.median(v)) if v else float("nan")
    res = dict(runs[-1])
    res.update({
        "available": True,
        "ttft_ms": med("ttft_ms"), "ttft_ms_best": best_lo("ttft_ms"),
        "decode_tok_s": med("decode_tok_s"), "decode_tok_s_best": best_hi("decode_tok_s"),
        "e2e_ms": med("e2e_ms"), "e2e_ms_best": best_lo("e2e_ms"),
        "gpu_pct": agg(gpu_pk), "mem_free_pct": agg(free_lo), "swap_mb": agg(swap_pk),
    })
    res["logits"] = np.fromfile(os.path.join(out_dir, res["logits_file"]), dtype=np.float32).reshape(3, -1)
    return res


def run_cute_test(onnx, out_dir):
    return _subrun("run_cute.py", onnx, out_dir, label="cute")


def run_luminal_test(onnx, out_dir):
    return _subrun("run_luminal.py", onnx, out_dir, label="luminal")


def run_burn_test(onnx, out_dir):
    return _subrun("run_burn.py", onnx, out_dir, label="burn")


def run_ort_cpu_test(onnx, out_dir):
    return _subrun("run_ort.py", onnx, out_dir, extra=["CPUExecutionProvider"], label="onnxruntime-cpu")


def run_ort_coreml_test(onnx, out_dir):
    return _subrun("run_ort.py", onnx, out_dir, extra=["CoreMLExecutionProvider"], label="onnxruntime-coreml")


def run_mlx_test(onnx, out_dir):
    return _subrun("run_mlx.py", onnx, out_dir, label="mlx")


def commit_results(results, out_dir):
    ref = next((r for r in results if r.get("available") and "logits" in r), None)
    for r in results:
        if not r.get("available") or "logits" not in r or ref is None:
            r["logit_check"] = "n/a"
            continue
        d = np.abs(r["logits"] - ref["logits"]).max()
        peak = np.abs(ref["logits"]).max()
        r["logit_check"] = "pass" if d <= FAIRNESS_REL_TOL * peak else f"FAIL({d/peak:.1e})"

    cols = ["name", "platform", "ttft_ms", "ttft_ms_best", "decode_tok_s", "decode_tok_s_best", "e2e_ms", "e2e_ms_best", "gpu_pct", "mem_free_pct", "swap_mb", "logit_check"]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r.get(c, "n/a" if not r.get("available") else "") for c in cols])
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    for r in results:
        with open(os.path.join(out_dir, "raw", f"{r['name']}.json"), "w") as f:
            json.dump({k: v for k, v in r.items() if k != "logits"}, f, indent=2)
    _chart(results, out_dir)
    print(f"\nresults -> {out_dir}")
    print(f"{'runner':<14}{'ttft(ms)':>10}{'tok/s':>8}{'e2e(ms)':>10}{'gpu%':>7}{'free%':>7}{'swap':>8}  logits")
    for r in results:
        if r.get("available"):
            print(f"{r['name']:<14}{r['ttft_ms']:>10.1f}{r['decode_tok_s']:>8.0f}{r['e2e_ms']:>10.1f}{r['gpu_pct']:>7.0f}{r['mem_free_pct']:>7.0f}{r['swap_mb']:>8.0f}  {r['logit_check']}")
        else:
            print(f"{r['name']:<14}{'(unavailable)':>28}")


def _chart(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    avail = [r for r in results if r.get("available")]
    if not avail:
        return
    names = [r["name"] for r in avail]
    fig, ax = plt.subplots(1, 5, figsize=(21, 4))
    for i, (key, title, lower) in enumerate([("ttft_ms", "prompt time (ms)", True), ("decode_tok_s", "decode tok/s", False), ("e2e_ms", "end-to-end (ms)", True)]):
        vals = [r[key] for r in avail]
        bars = ax[i].bar(names, vals, color="#4c6ef5")
        best = min(vals) if lower else max(vals)
        for b, v in zip(bars, vals):
            if v == best:
                b.set_color("#2f9e44")
        ax[i].set_title(title)
        ax[i].tick_params(axis="x", rotation=20)
    gvals = [r.get("gpu_pct", float("nan")) for r in avail]
    ax[3].bar(names, gvals, color="#f08c00")
    ax[3].set_title("peak GPU util (%, system-wide)")
    ax[3].set_ylim(0, 100)
    ax[3].tick_params(axis="x", rotation=20)
    fvals = [r.get("mem_free_pct", float("nan")) for r in avail]
    svals = [r.get("swap_mb", float("nan")) for r in avail]
    ax[4].bar(names, fvals, color="#1098ad")
    ax[4].set_title("min mem free (%) & peak swap (MB)")
    ax[4].set_ylim(0, 100)
    ax[4].set_ylabel("mem free %", color="#1098ad")
    ax[4].tick_params(axis="x", rotation=20)
    axs = ax[4].twinx()
    axs.plot(names, svals, color="#e03131", marker="o", linestyle="none")
    axs.set_ylabel("swap MB", color="#e03131")
    axs.set_ylim(bottom=0)
    fig.suptitle(f"fat model  P={P} G={G} C={C}  (green=best; GPU/mem are system-wide, best-effort)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chart.png"), dpi=110)


def main():
    onnx = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models/fat/model.onnx")
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(ROOT, "results", f"profile_test_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    results = [
        run_cute_test(onnx, out_dir),
        run_luminal_test(onnx, out_dir),
        run_burn_test(onnx, out_dir),
        run_mlx_test(onnx, out_dir),
        run_ort_coreml_test(onnx, out_dir),
        run_ort_cpu_test(onnx, out_dir),
    ]
    commit_results(results, out_dir)


if __name__ == "__main__":
    main()
