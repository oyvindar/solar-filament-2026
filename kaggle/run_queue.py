"""Run Kaggle kernels through the 2-concurrent-GPU-session limit and download their outputs.

Usage: python kaggle/run_queue.py <out_root> <kernel_dir> [<kernel_dir> ...]
Kernels whose slug is already running/complete on Kaggle are not re-pushed; a kernel dir
is pushed when a GPU slot is free. Needs KAGGLE_API_TOKEN and the kaggle CLI importable.
"""
import json
import os
import subprocess
import sys
import time

USER = "saudsaudesen"
MAX_GPU = 2


def kg(*args, timeout=900):
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def status(slug):
    _, out = kg("kernels", "status", f"{USER}/{slug}")
    low = out.lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in low:
            return s
    return "unknown"


def main(out_root, dirs, poll=60, max_hours=11.5):
    t0 = time.time()
    stamp = lambda: f"[{(time.time() - t0) / 60:6.1f} min]"
    todo = []  # (slug, dir, pushed)
    for d in dirs:
        slug = json.load(open(os.path.join(d, "kernel-metadata.json")))["id"].split("/")[1]
        st = status(slug)
        todo.append([slug, d, st in ("running", "queued", "complete", "error")])
        print(stamp(), slug, "initial status:", st, flush=True)
    done = set()
    while len(done) < len(todo) and time.time() - t0 < max_hours * 3600:
        running = 0
        for item in todo:
            slug, d, pushed = item
            if slug in done or not pushed:
                continue
            st = status(slug)
            if st in ("complete", "error", "cancel"):
                dest = os.path.join(out_root, slug)
                os.makedirs(dest, exist_ok=True)
                rc, out = kg("kernels", "output", f"{USER}/{slug}", "-p", dest, "-o", timeout=3600)
                n = sum(len(f) for _, _, f in os.walk(dest))
                print(stamp(), slug, st, f"| downloaded {n} files (rc {rc})", flush=True)
                done.add(slug)
            else:
                running += 1
        for item in todo:
            slug, d, pushed = item
            if not pushed and running < MAX_GPU:
                rc, out = kg("kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4")
                ok = "successfully pushed" in out
                print(stamp(), slug, "push:", out.strip()[-120:], flush=True)
                if ok:
                    item[2] = True
                    running += 1
        if len(done) < len(todo):
            time.sleep(poll)
    print(stamp(), "finished:", sorted(done), "| not finished:", [t[0] for t in todo if t[0] not in done], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
