"""Recover an interrupted public encoder download and verify the official LFS hash.

Uses metadata already fetched from the official Hub. No pickle files are loaded.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODEL = "BAAI/bge-small-en-v1.5"


def main():
    cache = Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5"
    revision = (cache / "refs/main").read_text().strip()
    tree = json.loads((cache / "trees" / f"{revision}.json").read_text())
    info = tree["files"]["model.safetensors"]
    expected, size = info["lfs_sha256"], info["size"]
    partial = ROOT / "runs/model_download/model.safetensors.part"
    partial.parent.mkdir(parents=True, exist_ok=True)
    if not partial.exists():
        candidates = list((cache / "blobs").glob(expected + "*.incomplete"))
        if candidates:
            shutil.copyfile(max(candidates, key=lambda p: p.stat().st_size), partial)
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            digest = hashlib.sha256(partial.read_bytes()).hexdigest()
            if digest == expected:
                dest = cache / "snapshots" / revision / "model.safetensors"
                shutil.copyfile(partial, dest)
                print(json.dumps({"verified": True, "sha256": digest, "bytes": size}), flush=True)
                return
            # Preserve rejected bytes for inspection and restart without trusting a corrupt prefix.
            partial.rename(partial.with_suffix(f".rejected-{attempt}"))
            offset = 0
        host = ("https://huggingface.co", "https://hf-mirror.com")[attempt % 2]
        url = f"{host}/{MODEL}/resolve/{revision}/model.safetensors?download=true"
        try:
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30, connect=15)) as client:
                with client.stream("GET", url, headers={"Range": f"bytes={offset}-"}) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        start = int(response.headers["content-range"].split()[1].split("-")[0])
                        if start != offset:
                            raise ValueError("Unexpected range start")
                        mode = "ab"
                    else:
                        mode = "wb"
                    with partial.open(mode) as stream:
                        for chunk in response.iter_bytes(1024*1024):
                            stream.write(chunk)
            print(f"download attempt {attempt+1}: {partial.stat().st_size}/{size} bytes", flush=True)
        except (httpx.HTTPError, ValueError) as exc:
            print(f"download attempt {attempt+1}: {type(exc).__name__}", flush=True)
    raise RuntimeError("Encoder download incomplete; no unverified weights installed")


if __name__ == "__main__":
    main()
