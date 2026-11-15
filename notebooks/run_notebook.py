"""Run a notebook's code cells in order in one Python process ("Run All") and store their outputs.

Usage: python notebooks/run_notebook.py notebooks/end_to_end.ipynb notebooks/end_to_end.ipynb
Used where a Jupyter kernel cannot be started (no local sockets). Text printed by a cell becomes a
stream output; every matplotlib figure open at the end of a cell becomes a PNG display output.
The notebook is saved after every cell, so a failure keeps the outputs of the cells before it.
"""
import base64
import contextlib
import io
import json
import sys
import time
import traceback


def run(path_in: str, path_out: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in [m for m in sys.modules if m == "fsc" or m.startswith("fsc.")]:
        del sys.modules[name]  # always run the current code, like a fresh kernel
    nb = json.load(open(path_in, encoding="utf-8"))
    ns = {"__name__": "__notebook__"}
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        n += 1
        src = "".join(cell["source"])
        buf, t0, err = io.StringIO(), time.time(), None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(src, f"<cell {n}>", "exec"), ns)
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
        outputs = []
        if buf.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
        for num in plt.get_fignums():
            png = io.BytesIO()
            plt.figure(num).savefig(png, format="png", dpi=110, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(png.getvalue()).decode("ascii"), "text/plain": ["<Figure>"]}})
        plt.close("all")
        if err:
            lines = err.rstrip().splitlines()
            outputs.append({"output_type": "error", "ename": lines[-1].split(":")[0], "evalue": lines[-1], "traceback": lines})
        cell["outputs"], cell["execution_count"] = outputs, n
        with open(path_out, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
        print(f"cell {n}: {time.time() - t0:.0f}s {'ERROR' if err else 'ok'}", flush=True)
        if err:
            print(err, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1], sys.argv[2]))
