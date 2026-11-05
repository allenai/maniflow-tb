#!/usr/bin/env python3
"""
Batch re-encode RGB videos to 224x224 with the dataloader's PIL bilinear.

The resize is the same two Pillow lines as
``hvla.dataloader.franka_dataset._pil_to_tensor`` (see ``pil_bilinear.py``).
It is applied to decoded uint8 RGB *before* the encoder, so the frames
handed to ffmpeg are bit-exact with what training would resize today.

Output layout mirrors the source, with a ``_<size>`` suffix:

    <SRC_ROOT>/observation.images.camera:front_camera/chunk-XXX/file-YYY.mp4
    <SRC_ROOT>/observation.images.camera:front_camera_224/chunk-XXX/file-YYY.mp4

Default encode is lossy SVT-AV1 (same codec family as the 640x360 sources).
Use ``--codec lossless`` for libx264rgb qp=0 if you need a bit-exact file
(much larger; see README).

Writes are atomic and existing outputs are skipped, so the job is resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pil_bilinear import dataloader_pil_bilinear_ref, resize_rgb_uint8

IMAGE_PREFIX = "observation.images."
DEFAULT_CAMS = [
    "observation.images.camera:front_camera",
    "observation.images.camera:gripper_camera",
]


def resolve_videos_root(path):
    """Accept either the dataset root or its ``videos/`` directory."""
    if os.path.isdir(os.path.join(path, "videos")):
        names = os.listdir(path)
        if not any(n.startswith(IMAGE_PREFIX) for n in names):
            return os.path.join(path, "videos")
    return path


def list_rgb_cams(src_root, size, cams=None):
    suffix = f"_{size}"
    if cams:
        return list(cams)
    found = []
    for name in sorted(os.listdir(src_root)):
        if not name.startswith(IMAGE_PREFIX):
            continue
        if name.endswith(suffix):
            continue
        if os.path.isdir(os.path.join(src_root, name)):
            found.append(name)
    return found or list(DEFAULT_CAMS)


def chunk_index(name):
    """Parse ``chunk-010`` -> 10. Returns None if the name is not a chunk dir."""
    if not name.startswith("chunk-"):
        return None
    try:
        return int(name.split("-", 1)[1])
    except ValueError:
        return None


def build_jobs(src_root, cams, suffix, skip_existing=True, first_n_chunks=None):
    jobs = []
    for cam in cams:
        srcdir = os.path.join(src_root, cam)
        if not os.path.isdir(srcdir):
            continue
        for chunk in sorted(os.listdir(srcdir)):
            cdir = os.path.join(srcdir, chunk)
            if not os.path.isdir(cdir):
                continue
            idx = chunk_index(chunk)
            if first_n_chunks is not None and (idx is None or idx >= first_n_chunks):
                continue
            for fn in sorted(os.listdir(cdir)):
                if not fn.endswith(".mp4"):
                    continue
                src = os.path.join(cdir, fn)
                dst = os.path.join(src_root, cam + suffix, chunk, fn)
                if skip_existing and os.path.exists(dst):
                    continue
                jobs.append((src, dst))
    return jobs


def probe(src):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json", src,
        ],
        capture_output=True, text=True, check=True,
    )
    st = json.loads(out.stdout)["streams"][0]
    return int(st["width"]), int(st["height"]), st.get("r_frame_rate", "25/1")


def encode_argv(dst, size, fps, codec, crf, preset, gop):
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-threads", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{size}x{size}", "-r", str(fps), "-i", "-",
        "-an",
    ]
    if codec == "lossless":
        cmd += [
            "-c:v", "libx264rgb", "-qp", "0", "-g", "1",
            "-preset", preset, "-pix_fmt", "rgb24",
        ]
    elif codec == "av1":
        cmd += ["-c:v", "libsvtav1", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
        if gop is not None:
            cmd += ["-g", str(gop)]
    else:
        raise ValueError(f"unknown codec {codec!r}")
    cmd.append(dst)
    return cmd


def process(args):
    src, dst, size, codec, crf, preset, gop = args
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        w, h, fps = probe(src)
        frame_bytes = w * h * 3
        dec = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-threads", "1", "-i", src,
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            stdout=subprocess.PIPE,
        )
        tmp = dst + ".tmp.mp4"
        enc = subprocess.Popen(
            encode_argv(tmp, size, fps, codec, crf, preset, gop),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        n = 0
        try:
            while True:
                buf = dec.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
                if not frame.flags.writeable:
                    frame = np.array(frame, copy=True)
                small = resize_rgb_uint8(frame, size)
                if n == 0:
                    ref = dataloader_pil_bilinear_ref(frame, size)
                    if not np.array_equal(small, ref):
                        raise RuntimeError("PIL bilinear self-check failed on frame 0")
                enc.stdin.write(np.ascontiguousarray(small).tobytes())
                n += 1
            enc.stdin.close()
            enc_stderr = enc.stderr.read()
            enc.wait()
            dec.wait()
            if enc.returncode != 0:
                if os.path.exists(tmp):
                    os.remove(tmp)
                return (dst, False, enc_stderr.decode()[-300:])
            if n == 0:
                if os.path.exists(tmp):
                    os.remove(tmp)
                return (dst, False, "decoded 0 frames")
            os.replace(tmp, dst)
            return (dst, True, f"{n}f {w}x{h}->{size}")
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            dec.kill()
            enc.kill()
            raise
    except subprocess.CalledProcessError as e:
        return (dst, False, (e.stderr or b"").decode()[-300:])
    except Exception as e:  # noqa
        return (dst, False, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True,
                    help="dataset root or videos/ directory holding observation.images.*")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--codec", choices=("av1", "lossless"), default="av1")
    ap.add_argument("--crf", type=int, default=26,
                    help="SVT-AV1 CRF (ignored for lossless). 26 is the training default; "
                         "not comparable to x264 CRF 23 (that maps to ~SVT-AV1 30).")
    ap.add_argument("--preset", default=None,
                    help="svt-av1 speed preset 0-13 (default 8, faster) or x264 "
                         "preset (default medium). Higher SVT preset = faster, slightly "
                         "less efficient at the same CRF.")
    ap.add_argument("--gop", type=int, default=None)
    ap.add_argument("--cams", nargs="*", default=None,
                    help="camera directory names; default = all observation.images.* "
                         "except already-converted _<size> dirs")
    ap.add_argument("--no-skip", action="store_true",
                    help="re-encode even if destination exists")
    ap.add_argument("--limit", type=int, default=0,
                    help="only encode the first N files (0 = all); for pilot runs")
    ap.add_argument("--first-n-chunks", type=int, default=None,
                    help="only encode chunk-000 .. chunk-(N-1), e.g. 11 -> through chunk-010")
    ap.add_argument("--progress-every", type=float, default=120,
                    help="print a progress line at least every this many seconds (also every 200 files)")
    ap.add_argument("--fail-log", default=None)
    args = ap.parse_args()

    if args.preset is None:
        args.preset = "medium" if args.codec == "lossless" else "8"

    src_root = resolve_videos_root(args.src_root)
    suffix = f"_{args.size}"
    cams = list_rgb_cams(src_root, args.size, args.cams)
    fail_log = args.fail_log or os.path.join(
        os.path.dirname(src_root.rstrip("/")), f"reencode_rgb{suffix}_failures.log")
    jobs = build_jobs(
        src_root, cams, suffix,
        skip_existing=not args.no_skip,
        first_n_chunks=args.first_n_chunks,
    )
    if args.limit:
        jobs = jobs[:args.limit]
    total = len(jobs)
    chunk_note = (f"first_n_chunks={args.first_n_chunks} "
                  if args.first_n_chunks is not None else "")
    print(
        f"[plan] {total} files, cams={cams} -> <camera>{suffix}, {chunk_note}"
        f"PIL BILINEAR {args.size}x{args.size}, codec={args.codec} "
        f"crf={args.crf} preset={args.preset} workers={args.workers} "
        f"skip_existing={not args.no_skip}",
        flush=True,
    )
    if total == 0:
        print("[plan] nothing to do.", flush=True)
        return

    payload = [
        (s, d, args.size, args.codec, args.crf, args.preset, args.gop) for (s, d) in jobs
    ]
    ok = fail = 0
    t0 = time.time()
    last_print = t0
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, job) for job in payload]
        for i, fut in enumerate(as_completed(futs), 1):
            dst, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                failures.append(f"{dst}\t{msg}")
            now = time.time()
            if i % 200 == 0 or i == total or (now - last_print) >= args.progress_every:
                el = now - t0
                rate = i / el if el else 0
                eta = (total - i) / rate if rate else 0
                print(f"[{i}/{total}] ok={ok} fail={fail} "
                      f"{rate:.2f} files/s elapsed={el:.0f}s eta={eta:.0f}s",
                      flush=True)
                last_print = now

    if failures:
        with open(fail_log, "w") as f:
            f.write("\n".join(failures) + "\n")
        print(f"[done] ok={ok} fail={fail}  failures logged -> {fail_log}", flush=True)
    else:
        print(f"[done] ok={ok} fail={fail}  (no failures)", flush=True)


if __name__ == "__main__":
    main()
