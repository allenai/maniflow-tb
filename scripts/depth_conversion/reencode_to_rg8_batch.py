#!/usr/bin/env python3
"""
Batch re-encode uint12 (gray12le) depth videos into 8-bit RG8 videos.

Packs (--pack), B always 0:

  8+4 (default)  R = mm & 0xFF, G = mm >> 8,      unpack R + G*256     (12-bit, 0..4095)
  5+5            R = mm >> 5,   G = mm & 31,      unpack (R << 5) | G  (10-bit, clip 1023)

Lossless (libx264rgb, crf=0), GOP=1 (every frame an intra keyframe), mp4.

For 8+4 native, a gray12le raw frame is little-endian 16-bit words, so the
bytes are already [low, high] per pixel and need no bit arithmetic.

Two output sizes, selected with --out-size:

  native (default)  keep 640x360, write to <camera>_rg8 or <camera>_rg8_55
  224               nearest-downscale to 224x224 first, write to <camera>_rg8_224
                    or <camera>_rg8_55_224

The 224 mode moves the resize the training dataloader already performs into the
stored file, which is what makes it ~3.4x smaller and ~3.4x faster to load. It
reproduces exactly what hvla/dataloader/franka_dataset.py does in
`_decode_depth_raw`:

    Image.fromarray(depth_m).resize((224, 224), Image.NEAREST)   # depth_m is float32

Two details make that delicate:

* Pillow's NEAREST index mapping is *not* the same for every mode. For float32 ("F")
  and int32 ("I") input it accumulates the source coordinate in a loop, so wherever
  (i + 0.5) * src/dst lands on an exact integer it drifts one pixel low; for uint16
  ("I;16") it lands on the arithmetic floor((i + 0.5) * src/dst). Over 640 -> 224
  that is 20 of the 224 output columns. The dataloader hands PIL a float32 array, so
  "F" is the mapping to preserve — hence the maps are read out of PIL itself
  (`nearest_index_maps`) rather than computed with floor().
* The dataloader masks (`depth_mm == 0` and `depth_m > MAX_DEPTH_M` -> 0) before it
  resizes. Those are per-pixel ops and NEAREST only ever selects source pixels, so
  masking commutes with the downscale and storing unmasked millimetres is safe.

Sources (uint12 originals):
  <SRC_ROOT>/observation.depth.camera:front_camera/chunk-XXX/file-YYY.mp4
  <SRC_ROOT>/observation.depth.camera:gripper_camera/chunk-XXX/file-YYY.mp4
Destinations (mirror layout, with an _rg8 / _rg8_224 suffix):
  <SRC_ROOT>/observation.depth.camera:front_camera_rg8/chunk-XXX/file-YYY.mp4
  <SRC_ROOT>/observation.depth.camera:gripper_camera_rg8/chunk-XXX/file-YYY.mp4
"""

import os
import time
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

W, H = 640, 360
OUTPUT_SUFFIX = "_rg8"
CAMS = [
    "observation.depth.camera:front_camera",
    "observation.depth.camera:gripper_camera",
]
FRAME_BYTES = H * W * 2  # gray12le: 2 bytes/pixel
RG8_55_MAX_MM = 1023  # 10-bit


def pack_to_rgb(depth_mm, pack):
    rgb = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    if pack == "5+5":
        mm = np.minimum(depth_mm, RG8_55_MAX_MM)
        rgb[..., 0] = (mm >> 5).astype(np.uint8)
        rgb[..., 1] = (mm & np.uint16(31)).astype(np.uint8)
    else:
        rgb[..., 0] = (depth_mm & 0xFF).astype(np.uint8)
        rgb[..., 1] = (depth_mm >> 8).astype(np.uint8)
    return rgb


def suffix_for(pack, out_size):
    tag = "_55" if pack == "5+5" else ""
    if out_size == "native":
        return f"{OUTPUT_SUFFIX}{tag}"
    return f"{OUTPUT_SUFFIX}{tag}_{int(out_size)}"


def nearest_index_maps(size, src_h=H, src_w=W):
    """Source row/column indices Pillow picks for a NEAREST resize to (size, size).

    Read out of Pillow by resizing coordinate ramps in mode "F", the mode the
    dataloader resizes in. The mapping is separable (rows depend only on the output
    row, columns only on the output column), so one ramp per axis captures it.
    """
    from PIL import Image

    rows = np.repeat(np.arange(src_h, dtype=np.float32)[:, None], src_w, axis=1)
    cols = np.repeat(np.arange(src_w, dtype=np.float32)[None, :], src_h, axis=0)
    ys = np.array(Image.fromarray(rows).resize((size, size), Image.NEAREST))
    xs = np.array(Image.fromarray(cols).resize((size, size), Image.NEAREST))
    assert np.all(ys == ys[:, :1]) and np.all(xs == xs[:1, :]), \
        "PIL NEAREST mapping is not separable; cannot reduce it to per-axis indices"
    return ys[:, 0].astype(np.intp), xs[0, :].astype(np.intp)


def chunk_index(name):
    if not name.startswith("chunk-"):
        return None
    try:
        return int(name.split("-", 1)[1])
    except ValueError:
        return None


def build_jobs(src_root, suffix=OUTPUT_SUFFIX, skip_existing=True, first_n_chunks=None, cams=None):
    jobs = []
    for cam in (cams or CAMS):
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


def process(args):
    src, dst, preset, maps, pack = args
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # decode native 12-bit (no rescale); bytes are [low, high] per pixel
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-threads", "1", "-i", src,
             "-f", "rawvideo", "-pix_fmt", "gray12le", "-"],
            capture_output=True, check=True,
        ).stdout
        n = len(raw) // FRAME_BYTES
        if n == 0 or len(raw) % FRAME_BYTES != 0:
            return (dst, False, f"unexpected byte count {len(raw)}")
        if maps is None and pack != "5+5":
            out_w, out_h = W, H
            b = np.frombuffer(raw, dtype=np.uint8).reshape(n, H, W, 2)
            rgb = np.zeros((n, H, W, 3), dtype=np.uint8)
            rgb[..., 0:2] = b  # R=low, G=high, B stays 0
        else:
            depth_mm = np.frombuffer(raw, dtype="<u2").reshape(n, H, W)
            if maps is not None:
                ys, xs = maps
                depth_mm = depth_mm[:, ys][:, :, xs]
                out_w = out_h = len(ys)
            else:
                out_w, out_h = W, H
            rgb = pack_to_rgb(depth_mm, pack)
            del depth_mm
        del raw

        tmp = dst + ".tmp.mp4"
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-threads", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-r", "25",
             "-i", "-", "-c:v", "libx264rgb", "-crf", "0", "-g", "1",
             "-preset", preset, "-pix_fmt", "rgb24", tmp],
            input=rgb.tobytes(), capture_output=True,
        )
        if p.returncode != 0:
            if os.path.exists(tmp):
                os.remove(tmp)
            return (dst, False, p.stderr.decode()[-300:])
        os.replace(tmp, dst)  # atomic -> skip-existing on resume is safe
        return (dst, True, f"{n}f")
    except subprocess.CalledProcessError as e:
        return (dst, False, (e.stderr or b"").decode()[-300:])
    except Exception as e:  # noqa
        return (dst, False, repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True,
                    help="root containing the original camera directories")
    ap.add_argument("--workers", type=int, default=192)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--out-size", default="native",
                    help="'native' for 640x360 -> <camera>_rg8[_55], or a square size such "
                         "as 224 for a PIL-identical nearest downscale -> <camera>_rg8[_55]_224")
    ap.add_argument("--pack", default="8+4", choices=["8+4", "5+5"],
                    help="RG8 bit split. 5+5 clips to 1023 mm and writes <camera>_rg8_55[_size]")
    ap.add_argument("--no-skip", action="store_true",
                    help="re-encode even if destination exists")
    ap.add_argument("--limit", type=int, default=0,
                    help="only encode the first N files (0 = all); for pilot runs")
    ap.add_argument("--first-n-chunks", type=int, default=None,
                    help="only encode chunk-000 .. chunk-(N-1), e.g. 11 -> through chunk-010")
    ap.add_argument("--progress-every", type=float, default=120,
                    help="print a progress line at least every this many seconds")
    ap.add_argument("--cams", nargs="+", default=None,
                    help="camera directory names (default: front + gripper depth)")
    ap.add_argument("--fail-log", default=None,
                    help="failure log path (default: alongside src-root)")
    args = ap.parse_args()

    suffix = suffix_for(args.pack, args.out_size)
    if args.out_size == "native":
        maps, size_note = None, f"{W}x{H} (native, pack={args.pack})"
    else:
        maps = nearest_index_maps(int(args.out_size))
        size_note = (f"{args.out_size}x{args.out_size} nearest pack={args.pack}, "
                     f"rows {maps[0][:3].tolist()}... cols {maps[1][:3].tolist()}...")

    fail_log = args.fail_log or os.path.join(
        os.path.dirname(args.src_root), f"reencode{suffix}_failures.log")
    jobs = build_jobs(
        args.src_root, suffix=suffix, skip_existing=not args.no_skip,
        first_n_chunks=args.first_n_chunks, cams=args.cams,
    )
    if args.limit:
        jobs = jobs[:args.limit]
    total = len(jobs)
    chunk_note = (f" first_n_chunks={args.first_n_chunks}"
                  if args.first_n_chunks is not None else "")
    cam_note = f" cams={args.cams or CAMS}"
    print(f"[plan] {total} files to encode -> {size_note}, dirs <camera>{suffix} "
          f"(preset={args.preset}, workers={args.workers}, skip_existing={not args.no_skip}"
          f"{chunk_note}{cam_note})",
          flush=True)
    if total == 0:
        print("[plan] nothing to do.", flush=True)
        return

    payload = [(s, d, args.preset, maps, args.pack) for (s, d) in jobs]
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
                      f"{rate:.1f} files/s elapsed={el:.0f}s eta={eta:.0f}s",
                      flush=True)
                last_print = now

    if failures:
        with open(fail_log, "w") as f:
            f.write("\n".join(failures) + "\n")
        print(f"[done] ok={ok} fail={fail}  failures logged -> {fail_log}",
              flush=True)
    else:
        print(f"[done] ok={ok} fail={fail}  (no failures)", flush=True)


if __name__ == "__main__":
    main()
