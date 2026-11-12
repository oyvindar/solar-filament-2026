"""Resumable download of a Kaggle kernel's output files (skips files already on disk).

Usage: python kaggle/fetch_outputs.py <dest_root> <owner/slug> [<owner/slug> ...]
The kaggle CLI's `kernels output` re-requests every file; with several hundred
probability maps this trips connection resets. Here the listing is paged once
and only missing files are fetched, each with retries and backoff.
"""
import os
import sys
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def list_outputs(api, owner, slug, page_size=200, tries=6):
    items, token, log = [], "", None
    while True:
        for attempt in range(tries):
            try:
                with api.build_kaggle_client() as kaggle:
                    req = ApiListKernelSessionOutputRequest()
                    req.user_name, req.kernel_slug, req.page_size, req.page_token = owner, slug, page_size, token
                    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                break
            except Exception as e:  # noqa: BLE001
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"listing failed for {owner}/{slug}")
        items.extend(resp.files or [])
        log = log or resp.log
        token = resp.next_page_token
        if not token:
            return items, log


def fetch(kernel, dest_root, tries=6):
    owner, slug = kernel.split("/")
    dest = os.path.join(dest_root, slug)
    api = KaggleApi()
    api.authenticate()
    items, log = list_outputs(api, owner, slug)
    os.makedirs(dest, exist_ok=True)
    got = skipped = failed = 0
    for it in items:
        out = os.path.join(dest, it.file_name)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        for attempt in range(tries):
            try:
                r = requests.get(it.url, timeout=300)
                r.raise_for_status()
                with open(out + ".part", "wb") as fh:
                    fh.write(r.content)
                os.replace(out + ".part", out)
                got += 1
                break
            except Exception:  # noqa: BLE001
                time.sleep(5 * (attempt + 1))
        else:
            failed += 1
    if log:
        with open(os.path.join(dest, slug + ".log"), "w", encoding="utf-8") as fh:
            fh.write(log)
    print(f"{kernel}: listed {len(items)} | downloaded {got} | already had {skipped} | failed {failed}", flush=True)
    return failed


if __name__ == "__main__":
    bad = sum(fetch(k, sys.argv[1]) for k in sys.argv[2:])
    sys.exit(1 if bad else 0)
