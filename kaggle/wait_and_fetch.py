"""Poll Kaggle kernels until they finish, then download their outputs.

Usage: python kaggle/wait_and_fetch.py <out_root> <slug> [<slug> ...]
Needs KAGGLE_API_TOKEN in the environment and the `kaggle` CLI importable.
"""
import os
import subprocess
import sys
import time

USER = "saudsaudesen"


def kg(*args, timeout=900):
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def main(out_root, slugs, poll=45, max_hours=11.5):
    pending = list(slugs)
    t0 = time.time()
    while pending and time.time() - t0 < max_hours * 3600:
        for slug in list(pending):
            rc, out = kg("kernels", "status", f"{USER}/{slug}")
            low = out.lower()
            state = next((s for s in ("complete", "error", "cancel") if s in low), None)
            if state:
                dest = os.path.join(out_root, slug)
                os.makedirs(dest, exist_ok=True)
                rc2, out2 = kg("kernels", "output", f"{USER}/{slug}", "-p", dest, "-o")
                n = sum(len(f) for _, _, f in os.walk(dest))
                print(f"[{(time.time() - t0) / 60:6.1f} min] {slug}: {state} | downloaded {n} files (rc {rc2})", flush=True)
                pending.remove(slug)
        if pending:
            time.sleep(poll)
    if pending:
        print("still pending:", pending, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
