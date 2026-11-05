#!/usr/bin/env python3
"""
Batch re-encode RG8 depth videos into "g12": lossless HEVC gray12le, short GOP, no B-frames.

Why
---
RG8 (`libx264rgb -qp 0 -g 1`, R = low byte, G = high byte) is exact and fast to decode with
torchcodec, but every frame is an intra frame of a byte-split image: ~17 KB/frame (front camera)
and ~10 KB/frame (gripper) at 224x224. On a 37k-episode / 82M-frame dataset that is ~2.25 TB
per variant. The 12-bit depth compresses far better as what it is -- a 12-bit gray image with
temporal prediction:

    libx265  lossless=1  bframes=0  keyint=8  gray12le

Measured on a 224x224 depth dataset (20-episode pilot, warm page cache, random
access like training): front GOP 8 = 3.3 KB/frame and 2.8 ms/frame (RG8: 16.0 KB, 4.0 ms);
gripper GOP 4 = 5.2 KB/frame and 2.8 ms/frame (RG8: 10.4 KB, 3.3 ms). Bit-exact. On a cold
(evicted) file RG8 costs torchcodec 0.1-6 s per frame, g12 3-80 ms. `bframes=0` makes decode
order equal display order, so the dataloader seeks to the keyframe and decodes at most GOP
frames. The moving gripper view decodes slower per frame, hence the shorter GOP there
(`--gop-cam observation.depth.camera:gripper_camera=4`).

What it does
------------
For every `<src_root>/videos/<cam><src_suffix>/chunk-XXX/file-YYY.mp4` (default suffix
`_rg8_224`), decode RG8 -> uint16 mm (streamed through ffmpeg in blocks, ~50 MB RAM per worker),
assert max <= 4095, and encode to `<dst_root>/videos/<cam><dst_suffix>/chunk-XXX/file-YYY.mp4`
(default suffix `_g12_224`; `dst_root` defaults to `src_root`). Writes go to `.tmp.mp4` and are
renamed atomically; existing outputs are skipped, so the job is resumable and shardable
(`--shard I --num-shards N` slices the sorted file list `[I::N]`). `meta/info.json` is left
untouched (same convention as the RG8 bake: the dataloader picks the decoder from config, not
from the metadata tag). A small sidecar `meta/depth_<dst_suffix>.json` records codec, GOP,
preset and source suffix.

Examples
--------
    # pilot: 20 episodes of both cameras into a scratch root, GOP 8
    python3 scripts/depth_conversion/reencode_to_g12_batch.py \
        --src-root /path/to/dataset \
        --dst-root /tmp/g12_pilot --limit 20 --workers 8 --gop 8 --preset medium --verify-count

    # one shard of the full run (one CPU-only job per shard); GOP 8 for the front camera,
    # GOP 4 for the gripper camera
    python3 scripts/depth_conversion/reencode_to_g12_batch.py \
        --src-root /path/to/dataset \
        --gop 8 --gop-cam observation.depth.camera:gripper_camera=4 \
        --workers 32 --shard 3 --num-shards 16

`--bits 10` writes gray10le (clip 1023 mm) to `_g10_224`. `--crf 0.35` is lossy
(`_g10_224_crf035`); omit `--crf` for lossless. Source may be RG8 or gray12
(`--src-suffix ""`). `--out-size 224` nearest-downscales native gray12 first.

Train with `--depth_g12 g12_224` or `--depth_g12 g10_224` (mutually exclusive with `--depth_rg8`).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

CAMS = [
    "observation.depth.camera:front_camera",
    "observation.depth.camera:gripper_camera",
]
SRC_SUFFIX = "_rg8_224"
DST_SUFFIX = "_g12_224"
MAX12 = 4095
G10_MAX_MM = 1023
DEFAULT_GOP = 8
DEFAULT_PRESET = "medium"
BLOCK_FRAMES = 256
NATIVE_W, NATIVE_H = 640, 360


def chunk_index(name):
    if not name.startswith("chunk-"):
        return None
    try:
        return int(name.split("-", 1)[1])
    except ValueError:
        return None


def probe(path):
    """(width, height, r_frame_rate str, pix_fmt, codec_name, nb_frames or None) of the video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,pix_fmt,codec_name,nb_frames",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    st = json.loads(out)["streams"][0]
    nb = st.get("nb_frames")
    return (int(st["width"]), int(st["height"]), str(st["r_frame_rate"]), st.get("pix_fmt", ""),
            st.get("codec_name", ""), int(nb) if nb not in (None, "N/A") else None)


def count_frames(path):
    """Exact decoded frame count (ffprobe -count_frames), not the container's nb_frames."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0:nk=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out)


def videos_dir(root):
    v = os.path.join(root, "videos")
    return v if os.path.isdir(v) else root


def crf_tag(crf):
    return "" if crf is None else "_crf" + str(crf).replace(".", "")


def default_dst_suffix(bits, out_size, crf):
    parts = [f"g{bits}"]
    if out_size:
        parts.append(str(int(out_size)))
    elif bits == 12:
        parts.append("224")
    tag = crf_tag(crf)
    return "_" + "_".join(parts) + tag


def x265_params(gop, crf=None):
    # lossless: exact N-bit values. bframes=0: decode order == display order (cheap seek).
    # open-gop=0: IDR keyframes, no dependency across a GOP boundary. scenecut=0: keyframes
    # exactly every `gop` frames (predictable random-access cost). pools/frame-threads=1: one
    # thread per process; parallelism comes from the worker pool.
    common = (f"bframes=0:keyint={gop}:min-keyint={gop}:open-gop=0:scenecut=0:"
              f"pools=1:frame-threads=1:log-level=none")
    return f"lossless=1:{common}" if crf is None else f"crf={crf}:{common}"


def unpack_src_rgb(rgb, pack):
    r = rgb[..., 0].astype(np.uint16)
    g = rgb[..., 1].astype(np.uint16)
    return ((r << 5) | g) if pack == "5+5" else (r | (g << 8))


def nearest_index_maps(size, src_h=NATIVE_H, src_w=NATIVE_W):
    from PIL import Image
    rows = np.repeat(np.arange(src_h, dtype=np.float32)[:, None], src_w, axis=1)
    cols = np.repeat(np.arange(src_w, dtype=np.float32)[None, :], src_h, axis=0)
    ys = np.array(Image.fromarray(rows).resize((size, size), Image.NEAREST))
    xs = np.array(Image.fromarray(cols).resize((size, size), Image.NEAREST))
    return ys[:, 0].astype(np.intp), xs[0, :].astype(np.intp)


def build_jobs(src_root, dst_root, src_suffix, dst_suffix, cams, skip_existing=True,
               first_n_chunks=None):
    src_v, dst_v = videos_dir(src_root), videos_dir(dst_root)
    jobs = []
    for cam in cams:
        srcdir = os.path.join(src_v, cam + src_suffix)
        if not os.path.isdir(srcdir):
            print(f"[warn] source dir missing, skipped: {srcdir}", flush=True)
            continue
        for chunk in sorted(os.listdir(srcdir)):
            cdir = os.path.join(srcdir, chunk)
            if not os.path.isdir(cdir):
                continue
            idx = chunk_index(chunk)
            if first_n_chunks is not None and (idx is None or idx >= first_n_chunks):
                continue
            for fn in sorted(os.listdir(cdir)):
                if not fn.endswith(".mp4") or fn.endswith(".tmp.mp4"):
                    continue
                src = os.path.join(cdir, fn)
                dst = os.path.join(dst_v, cam + dst_suffix, chunk, fn)
                if skip_existing and os.path.exists(dst):
                    continue
                jobs.append((src, dst, cam))
    jobs.sort()
    return jobs


def gop_for(cam, args):
    """GOP for a camera: --gop-cam CAM=GOP overrides --gop."""
    for spec in args.gop_cam or []:
        k, v = spec.rsplit("=", 1)
        if k == cam:
            return int(v)
    return args.gop


def process(job):
    """Re-encode one file. Returns (dst, ok, message, n_frames, dst_bytes, seconds)."""
    src, dst, gop, preset, verify_count, bits, crf, out_size, src_pack = job
    t0 = time.time()
    tmp = dst + ".tmp.mp4"
    errlog = dst + ".tmp.log"
    dec = enc = None
    out_pix = f"gray{bits}le"
    max_mm = G10_MAX_MM if bits == 10 else MAX12
    try:
        w, h, fps, pix_fmt, codec, nb = probe(src)
        src_rg8 = codec == "h264" and pix_fmt in ("gbrp", "rgb24")
        src_gray = str(pix_fmt).startswith("gray")
        if not src_rg8 and not src_gray:
            return (dst, False, f"source is {codec}/{pix_fmt}, expected RG8 or gray*", 0, 0, time.time() - t0)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        dec_pix = "rgb24" if src_rg8 else ("gray10le" if str(pix_fmt).startswith("gray10") else "gray12le")
        bpp = 3 if src_rg8 else 2
        frame_bytes = w * h * bpp
        maps = None
        out_w, out_h = w, h
        if out_size and (w, h) != (out_size, out_size):
            maps = nearest_index_maps(out_size, h, w)
            out_w = out_h = out_size
        dec = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", src,
             "-f", "rawvideo", "-pix_fmt", dec_pix, "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        errf = open(errlog, "wb")
        enc_cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1",
            "-f", "rawvideo", "-pix_fmt", out_pix, "-s", f"{out_w}x{out_h}", "-r", fps, "-i", "pipe:0",
            "-c:v", "libx265", "-preset", preset,
        ]
        if crf is not None:
            enc_cmd += ["-crf", str(crf)]
        enc_cmd += [
            "-x265-params", x265_params(gop, crf), "-g", str(gop),
            "-pix_fmt", out_pix, "-an", "-movflags", "+faststart", "-f", "mp4", tmp,
        ]
        enc = subprocess.Popen(
            enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errf,
        )
        n = 0
        vmax = 0
        block = BLOCK_FRAMES * frame_bytes
        while True:
            buf = dec.stdout.read(block)
            if not buf:
                break
            k = len(buf) // frame_bytes
            if k * frame_bytes != len(buf):
                rest = dec.stdout.read(frame_bytes - (len(buf) - k * frame_bytes))
                buf += rest
                k = len(buf) // frame_bytes
                if k * frame_bytes != len(buf):
                    raise RuntimeError(f"decoder produced a partial frame ({len(buf)} bytes)")
            if src_rg8:
                rgb = np.frombuffer(buf, dtype=np.uint8).reshape(k, h, w, 3)
                depth = unpack_src_rgb(rgb, src_pack)
            else:
                depth = np.frombuffer(buf, dtype="<u2").reshape(k, h, w)
            if maps is not None:
                ys, xs = maps
                depth = depth[:, ys][:, :, xs]
            if bits == 10:
                depth = np.minimum(depth, G10_MAX_MM)
            m = int(depth.max())
            if m > vmax:
                vmax = m
            if vmax > max_mm:
                raise RuntimeError(f"depth max {vmax} > {max_mm}: does not fit {bits} bits")
            enc.stdin.write(depth.astype("<u2", copy=False).tobytes())
            n += k
        dec.stdout.close()
        rc_dec = dec.wait()
        enc.stdin.close()
        rc_enc = enc.wait()
        errf.close()
        if rc_dec != 0:
            raise RuntimeError(f"decoder exit {rc_dec}")
        if rc_enc != 0:
            msg = open(errlog, "rb").read().decode(errors="replace")[-300:]
            raise RuntimeError(f"encoder exit {rc_enc}: {msg}")
        if n == 0:
            raise RuntimeError("no frames decoded")
        if nb is not None and n != nb:
            raise RuntimeError(f"decoded {n} frames but source header says {nb}")
        if verify_count:
            n_out = count_frames(tmp)
            if n_out != n:
                raise RuntimeError(f"output has {n_out} frames, expected {n}")
        else:
            _, _, _, got_pix, out_codec, out_nb = probe(tmp)
            if out_codec != "hevc" or got_pix != out_pix:
                raise RuntimeError(f"output is {out_codec}/{got_pix}, expected hevc/{out_pix}")
            if out_nb is not None and out_nb != n:
                raise RuntimeError(f"output header has {out_nb} frames, expected {n}")
        os.replace(tmp, dst)
        size = os.path.getsize(dst)
        return (dst, True, f"{n}f {size / n / 1e3:.1f}KB/f max={vmax}", n, size, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        for proc in (dec, enc):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
        if os.path.exists(tmp):
            os.remove(tmp)
        return (dst, False, repr(e)[-400:], 0, 0, time.time() - t0)
    finally:
        if os.path.exists(errlog):
            try:
                os.remove(errlog)
            except OSError:
                pass


def write_sidecar(dst_root, args, cams):
    meta_dir = os.path.join(dst_root, "meta")
    if not os.path.isdir(meta_dir):
        return None
    path = os.path.join(meta_dir, f"depth{args.dst_suffix}.json")
    if os.path.exists(path):
        return path
    doc = {
        "variant_suffix": args.dst_suffix.lstrip("_"),
        "source_suffix": args.src_suffix.lstrip("_"),
        "cams": cams,
        "codec": "libx265",
        "pix_fmt": f"gray{args.bits}le",
        "x265_params": x265_params(args.gop, args.crf).replace(
            f"keyint={args.gop}:min-keyint={args.gop}", "keyint=<gop>:min-keyint=<gop>"),
        "gop": args.gop,
        "gop_per_cam": {cam: gop_for(cam, args) for cam in cams},
        "preset": args.preset,
        "depth_encoding": f"gray{args.bits}le_mm",
        "depth_max_mm": G10_MAX_MM if args.bits == 10 else MAX12,
        "bits": args.bits,
        "crf": args.crf,
        "lossless": args.crf is None,
        "bframes": 0,
        "tool": "scripts/depth_conversion/reencode_to_g12_batch.py",
        "date": time.strftime("%Y-%m-%d"),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", required=True,
                    help="dataset root (contains videos/ and meta/); sources are videos/<cam><src-suffix>")
    ap.add_argument("--dst-root", default=None,
                    help="where to write videos/<cam><dst-suffix> (default: the source root)")
    ap.add_argument("--src-suffix", default=SRC_SUFFIX)
    ap.add_argument("--dst-suffix", default=None,
                    help="default: _g12_224, or _g10[_SIZE][_crf035] when --bits 10")
    ap.add_argument("--bits", type=int, default=12, choices=[10, 12],
                    help="gray10 (clip 1023 mm) or gray12")
    ap.add_argument("--crf", type=float, default=None,
                    help="x265 CRF (e.g. 0.35). Omit for lossless")
    ap.add_argument("--out-size", type=int, default=None,
                    help="PIL-NEAREST square downscale before encode (e.g. 224)")
    ap.add_argument("--cams", nargs="+", default=None,
                    help="depth camera keys (default: front + gripper depth)")
    ap.add_argument("--gop", type=int, default=DEFAULT_GOP,
                    help="keyframe interval; random access decodes at most this many frames")
    ap.add_argument("--gop-cam", action="append", default=None, metavar="CAM=GOP",
                    help="per-camera GOP override, e.g. observation.depth.camera:gripper_camera=4 (repeatable)")
    ap.add_argument("--preset", default=DEFAULT_PRESET, help="x265 preset (speed/size trade-off)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--shard", type=int, default=None, help="shard index I of the sorted file list")
    ap.add_argument("--num-shards", type=int, default=1, help="number of shards N ([I::N])")
    ap.add_argument("--first-n-chunks", type=int, default=None,
                    help="only chunk-000 .. chunk-(N-1)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (0 = all); for pilots")
    ap.add_argument("--no-skip", action="store_true", help="re-encode even if the destination exists")
    ap.add_argument("--verify-count", action="store_true",
                    help="count decoded output frames with ffprobe -count_frames (slower, exact)")
    ap.add_argument("--progress-every", type=float, default=60)
    ap.add_argument("--fail-log", default=None,
                    help="failure log path (default: <dst-root>/meta/reencode<dst-suffix>_failures[_shardI].log)")
    ap.add_argument("--dry-run", action="store_true", help="list the work and exit")
    args = ap.parse_args()

    if args.shard is not None and not (0 <= args.shard < args.num_shards):
        ap.error("--shard must be in [0, --num-shards)")
    src_root = os.path.abspath(args.src_root)
    dst_root = os.path.abspath(args.dst_root) if args.dst_root else src_root
    cams = args.cams or CAMS
    if not args.src_suffix.startswith("_"):
        args.src_suffix = "_" + args.src_suffix if args.src_suffix else ""
    if args.dst_suffix is None:
        args.dst_suffix = default_dst_suffix(args.bits, args.out_size, args.crf)
    elif not args.dst_suffix.startswith("_"):
        args.dst_suffix = "_" + args.dst_suffix
    if src_root == dst_root and args.src_suffix == args.dst_suffix:
        ap.error("source and destination are the same directories")
    src_pack = "5+5" if "55" in args.src_suffix.strip("_").split("_") else "8+4"

    jobs = build_jobs(src_root, dst_root, args.src_suffix, args.dst_suffix, cams,
                      skip_existing=not args.no_skip, first_n_chunks=args.first_n_chunks)
    n_all = len(jobs)
    if args.shard is not None:
        jobs = jobs[args.shard::args.num_shards]
    if args.limit:
        jobs = jobs[:args.limit]
    total = len(jobs)
    shard_note = f" shard={args.shard}/{args.num_shards}" if args.shard is not None else ""
    gop_note = ", ".join(f"{c.split(':')[-1]}={gop_for(c, args)}" for c in cams)
    print(f"[plan] {total} files to encode (of {n_all} pending) -> {videos_dir(dst_root)}/<cam>{args.dst_suffix} "
          f"from <cam>{args.src_suffix}; bits={args.bits} crf={args.crf} out_size={args.out_size} "
          f"gop={gop_note} preset={args.preset} workers={args.workers} "
          f"skip_existing={not args.no_skip}{shard_note} cams={cams}", flush=True)
    if args.dry_run:
        for s, d, cam in jobs[:3]:
            print(f"  {s}\n    -> {d}  (gop {gop_for(cam, args)})")
        if total > 3:
            print(f"  ... {total - 3} more")
        return 0
    if total == 0:
        print("[plan] nothing to do.", flush=True)
        return 0

    if args.shard in (None, 0):
        side = write_sidecar(dst_root, args, cams)
        if side:
            print(f"[meta] sidecar: {side}", flush=True)

    fail_log = args.fail_log or os.path.join(
        dst_root if not os.path.isdir(os.path.join(dst_root, "meta")) else os.path.join(dst_root, "meta"),
        f"reencode{args.dst_suffix}_failures" + (f"_shard{args.shard}" if args.shard is not None else "") + ".log")

    payload = [(s, d, gop_for(cam, args), args.preset, args.verify_count,
                args.bits, args.crf, args.out_size, src_pack) for (s, d, cam) in jobs]
    ok = fail = 0
    frames = 0
    out_bytes = 0
    failures = []
    t0 = time.time()
    last_print = t0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, job) for job in payload]
        for i, fut in enumerate(as_completed(futs), 1):
            dst, success, msg, n, size, _sec = fut.result()
            if success:
                ok += 1
                frames += n
                out_bytes += size
            else:
                fail += 1
                failures.append(f"{dst}\t{msg}")
                print(f"[fail] {dst}: {msg}", flush=True)
            now = time.time()
            if i % 200 == 0 or i == total or (now - last_print) >= args.progress_every:
                el = now - t0
                rate = i / el if el else 0
                eta = (total - i) / rate if rate else 0
                kbf = out_bytes / frames / 1e3 if frames else 0
                print(f"[{i}/{total}] ok={ok} fail={fail} {rate:.2f} files/s "
                      f"{frames / el if el else 0:.0f} frames/s {kbf:.2f} KB/frame "
                      f"elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
                last_print = now

    if failures:
        os.makedirs(os.path.dirname(fail_log), exist_ok=True)
        with open(fail_log, "a") as f:
            f.write("\n".join(failures) + "\n")
        print(f"[done] ok={ok} fail={fail} frames={frames} bytes={out_bytes}  failures -> {fail_log}", flush=True)
        return 1
    print(f"[done] ok={ok} fail={fail} frames={frames} bytes={out_bytes} (no failures)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
