#!/usr/bin/env python3
"""
Compare 224 RGB videos against PIL-bilinear of the 640x360 originals.

The reference side is the array the dataloader would resize to, not ffmpeg
``scale``. After a lossy AV1 encode this will not be bit-exact; use
``--require-lossless`` when the files were written with ``--codec lossless``.

    python3 scripts/rgb_conversion/verify_reencode.py \
        --root /path/to/dataset/videos --samples 20 --frames 8 --full 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pil_bilinear import resize_rgb_uint8

IMAGE_PREFIX = "observation.images."


def resolve_videos_root(path):
    if os.path.isdir(os.path.join(path, "videos")):
        names = os.listdir(path)
        if not any(n.startswith(IMAGE_PREFIX) for n in names):
            return os.path.join(path, "videos")
    return path


def find_pairs(root, size):
    suffix = f"_{size}"
    pairs = []
    for name in sorted(os.listdir(root)):
        if not name.startswith(IMAGE_PREFIX) or name.endswith(suffix):
            continue
        recon_dir = os.path.join(root, name + suffix)
        orig_dir = os.path.join(root, name)
        if not os.path.isdir(recon_dir):
            continue
        for chunk in sorted(os.listdir(recon_dir)):
            cdir = os.path.join(recon_dir, chunk)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if not fn.endswith(".mp4"):
                    continue
                orig = os.path.join(orig_dir, chunk, fn)
                if os.path.exists(orig):
                    pairs.append((orig, os.path.join(cdir, fn)))
    return pairs


def probe_wh(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    st = json.loads(out.stdout)["streams"][0]
    nb = st.get("nb_frames")
    return int(st["width"]), int(st["height"]), int(nb) if nb and str(nb).isdigit() else None


def decode_rgb(path, w, h):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)


def writable(frame):
    return frame if frame.flags.writeable else np.array(frame, copy=True)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def compare_resized(orig, recon, size, idxs=None):
    """Return (ok, msg, max_abs). ``ok`` is always True unless shapes/counts fail."""
    ow, oh, _ = probe_wh(orig)
    rw, rh, _ = probe_wh(recon)
    if (rw, rh) != (size, size):
        return False, f"recon is {rw}x{rh}, expected {size}x{size}", None
    src = decode_rgb(orig, ow, oh)
    dst = decode_rgb(recon, rw, rh)
    if src.shape[0] != dst.shape[0]:
        return False, f"frame count mismatch orig={src.shape[0]} recon={dst.shape[0]}", None
    n = src.shape[0]
    if idxs is None:
        idxs = list(range(n))
    refs = np.stack([resize_rgb_uint8(writable(src[i]), size) for i in idxs])
    got = dst[np.asarray(idxs)]
    diff = np.abs(refs.astype(np.int16) - got.astype(np.int16))
    max_abs = int(diff.max())
    n_bad = int(np.count_nonzero(diff))
    msg = (
        f"{n}f, {len(idxs)} frames, max_abs={max_abs} "
        f"mean_abs={float(diff.mean()):.3f} mismatched_px={n_bad}/{refs.size} "
        f"psnr={psnr(refs, got):.2f}"
    )
    return True, msg, max_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root or videos/ directory")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--full", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-lossless", action="store_true",
                    help="fail unless every compared pixel is exact")
    args = ap.parse_args()

    root = resolve_videos_root(args.root)
    pairs = find_pairs(root, args.size)
    print(f"[scan] {len(pairs)} converted files with an original "
          f"(<cam> vs <cam>_{args.size})")
    if not pairs:
        print("[scan] nothing to verify.")
        return 1

    rng = random.Random(args.seed)
    sample = rng.sample(pairs, min(args.samples, len(pairs)))
    failures = []

    def consider(tag, orig, recon, idxs):
        try:
            ok, msg, max_abs = compare_resized(orig, recon, args.size, idxs)
        except Exception as e:  # noqa
            ok, msg, max_abs = False, repr(e), None
        if ok and args.require_lossless and max_abs != 0:
            ok = False
            msg = msg + "  (require-lossless)"
        print(f"{tag} {'PASS' if ok else 'FAIL'} {os.path.relpath(recon, root)}  {msg}",
              flush=True)
        if not ok:
            failures.append((recon, msg))

    for i, (orig, recon) in enumerate(sample, 1):
        ow, oh, n_orig = probe_wh(orig)
        n = n_orig or decode_rgb(orig, ow, oh).shape[0]
        idxs = sorted(set(np.linspace(0, n - 1, min(args.frames, n)).astype(int).tolist()))
        consider(f"[sampled {i}/{len(sample)}]", orig, recon, idxs)

    full = sample[: args.full] if args.full else []
    for i, (orig, recon) in enumerate(full, 1):
        consider(f"[full {i}/{len(full)}]", orig, recon, None)

    if failures:
        print(f"\n[done] {len(failures)} FAILURES:")
        for path, msg in failures:
            print(f"  {path}\t{msg}")
        return 1
    print(f"\n[done] all checks passed "
          f"({len(sample)} sampled x {args.frames} frames, {len(full)} full)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
