#!/usr/bin/env python3
"""
Verify that RG8-split re-encoded depth videos match their uint12 (gray12le)
originals bit-for-bit.

- ORIG: gray12le HEVC -> decoded with PyAV, read native 12-bit plane (uint16, 0..4095)
- RECON: RG8 lossless mp4 -> decoded with torchcodec (RGB).
  8+4 (default): value = R + G * 256
  5+5: value = (R << 5) | G, source clipped to 1023 mm before compare

Both decoders here are the ones the training dataloader actually uses
(hvla/dataloader/franka_dataset.py: _decode_depth_frame_16bit vs _decode_depth_rg8),
so a pass here means the dataloader will read identical depth either way.

Files are sampled from the re-encoded side, so this is safe to run mid-conversion.

    # 20 random files, 5 frames each, plus 2 whole-video bit-exact checks
    python3 verify_reencode.py --samples 20 --frames 5 --full 2

    # 5+5 pack (10-bit, clip 1023 mm); pack is inferred from --dst-suffix if omitted
    python3 verify_reencode.py --dst-suffix _rg8_55_224 --out-size 224 --pack 5+5

    # 224x224 downscaled RG8 (reencode_to_rg8_batch.py --out-size 224): the recon is a
    # different resolution than the original, so instead of comparing millimetres the
    # check runs the dataloader tail (mm -> metres, zero/2.5m masking, NEAREST resize
    # to 224) on both sides and requires the arrays the model sees to be identical
    python3 verify_reencode.py --root <dataset>/videos --src-suffix "" \
        --dst-suffix _rg8_224 --out-size 224

    # same check against the full-resolution RG8 set instead of the gray12le originals,
    # i.e. "does rg8_224 equal what the dataloader makes of rg8 today?"
    python3 verify_reencode.py --root <dataset>/videos --src-suffix _rg8 --src-rg8 \
        --dst-suffix _rg8_224 --out-size 224

    # files rebuilt by repair_corrupt_to_rg8.py hold zeros where the damaged source
    # never decoded, so they get their own check and are kept out of the sample
    python3 verify_reencode.py --repaired-list corrupt_files.txt

    # drop into pdb on the first sampled file (orig0 / recon0 in scope)
    python3 verify_reencode.py --interactive

Requires PyAV + torchcodec.
"""

import argparse
import os
import random
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = "/path/to/dataset/videos"  # override with --root
SRC_SUFFIX = "_unaltered"  # originals live in <cam>_unaltered, RG8 output in <cam>
CAMS = [
    "observation.depth.camera:front_camera",
    "observation.depth.camera:gripper_camera",
]

W, H = 640, 360
MAX_DEPTH_M = 2.5  # must match hvla/dataloader/franka_dataset.py
RG8_55_MAX_MM = 1023


def find_pairs(root=ROOT, src_suffix=SRC_SUFFIX, dst_suffix=""):
    """List (orig, recon) paths for every re-encoded file that has an original."""
    pairs = []
    for cam in CAMS:
        recon_dir = os.path.join(root, cam + dst_suffix)
        orig_dir = os.path.join(root, cam + src_suffix)
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


def suffix_parts(suffix):
    return suffix.strip("_").split("_") if suffix else []


def infer_pack(suffix):
    return "5+5" if "55" in suffix_parts(suffix) else "8+4"


def infer_bits(suffix):
    return 10 if "g10" in suffix_parts(suffix) or infer_pack(suffix) == "5+5" else 12


def infer_kind(suffix):
    head = (suffix_parts(suffix) or [""])[0]
    return "gray" if head.startswith("g10") or head.startswith("g12") else "rg8"


def infer_lossy(suffix):
    return any(p.startswith("crf") for p in suffix_parts(suffix))


def diff_msg(a, b, lossy):
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if not lossy:
        worst = float(diff.max()) if diff.size else 0.0
        if worst != 0:
            return False, f"max_abs_diff={worst} mismatched_px={int(np.count_nonzero(diff))}/{diff.size}"
        return True, "exact"
    mae = float(diff.mean()) if diff.size else 0.0
    p99 = float(np.percentile(diff, 99)) if diff.size else 0.0
    exact = float((diff == 0).mean()) if diff.size else 1.0
    return True, f"mae={mae:.3f} p99={p99:.1f} max={float(diff.max()) if diff.size else 0:.1f} exact={100 * exact:.1f}%"


def unpack_rg8(r, g, pack):
    r = r.astype(np.uint16)
    g = g.astype(np.uint16)
    return ((r << 5) | g) if pack == "5+5" else (r + g * 256)


def clip_src(depth_mm, bits):
    return np.minimum(depth_mm, RG8_55_MAX_MM) if bits == 10 else depth_mm


def dataloader_depth_m(depth_mm, size):
    """The tail of franka_dataset._decode_depth_raw: mm -> metres, mask, NEAREST resize.

    Reproduced verbatim so the comparison is against what training actually consumes.
    Frames already at (size, size) skip the resize, exactly as the dataloader does.
    """
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[depth_mm == 0] = 0.0
    depth_m[depth_m > MAX_DEPTH_M] = 0.0
    if depth_m.shape != (size, size):
        depth_m = np.array(Image.fromarray(depth_m).resize((size, size), Image.NEAREST))
    return depth_m


def nb_frames(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0:nk=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out)


def load_gray12_frames(path, idxs):
    """Decode the requested frame indices of a gray12le video with PyAV and read
    the native 12-bit plane directly (no reformat -> no bit-depth rescaling).

    The plane stores each 12-bit value in a little-endian 16-bit word; rows may
    be padded, so we reshape using the plane's line_size and crop to width."""
    import av

    want = set(idxs)
    out = {}
    with av.open(path) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i not in want:
                continue
            assert frame.format.name.startswith("gray"), (
                f"expected gray* source, got {frame.format.name}"
            )
            plane = frame.planes[0]
            buf = np.frombuffer(bytes(plane), dtype="<u2")
            stride = plane.line_size // 2      # uint16 elements per row (incl. padding)
            out[i] = buf.reshape(frame.height, stride)[:, :frame.width].copy()
            if len(out) == len(want):
                break
    return out


def load_rg8_frames(path, idxs, pack="8+4"):
    """Decode the requested frame indices of an RG8 video with torchcodec
    (RGB, uint8) and reconstruct depth. Also returns the max value
    seen on the unused B channel, which must stay 0."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(path)
    out = {}
    b_max = 0
    for i in idxs:
        rgb = decoder[i].cpu().numpy()      # (C, H, W) uint8, RGB
        out[i] = unpack_rg8(rgb[0], rgb[1], pack)
        b_max = max(b_max, int(rgb[2].max()))
    return out, b_max


def load_recon_frames(path, idxs, kind, pack):
    if kind == "gray":
        return load_gray12_frames(path, idxs), 0
    return load_rg8_frames(path, idxs, pack)


def check_sampled(orig, recon, n_frames, size=None, src_rg8=False, pack="8+4",
                  src_pack="8+4", kind="rg8", bits=12, lossy=False):
    """Compare n_frames evenly spaced frames. Returns (ok, message).

    Without --out-size the two sides share a resolution and raw millimetres are
    compared. With it the recon is smaller, so both sides go through the dataloader
    tail (which resizes the original and leaves the recon alone) and the resulting
    metre arrays must be identical. CRF / lossy dests report MAE instead of requiring exact.
    """
    n_orig, n_recon = nb_frames(orig), nb_frames(recon)
    if n_orig != n_recon:
        return False, f"frame count mismatch orig={n_orig} recon={n_recon}"

    idxs = sorted(set(np.linspace(0, n_orig - 1, min(n_frames, n_orig)).astype(int).tolist()))
    if src_rg8:
        o, o_b_max = load_rg8_frames(orig, idxs, src_pack)
        if o_b_max != 0:
            return False, f"source blue channel not zero (max={o_b_max})"
    else:
        o = load_gray12_frames(orig, idxs)
    r, b_max = load_recon_frames(recon, idxs, kind, pack)

    missing = [i for i in idxs if i not in o or i not in r]
    if missing:
        return False, f"could not decode frames {missing}"
    if b_max != 0:
        return False, f"blue channel not zero (max={b_max})"

    chunks_a, chunks_b = [], []
    for i in idxs:
        a, b = clip_src(o[i], bits), r[i]
        if size is not None:
            if b.shape != (size, size):
                return False, f"frame {i} recon is {b.shape}, expected ({size}, {size})"
            a, b = dataloader_depth_m(a, size), dataloader_depth_m(b, size)
        elif a.shape != b.shape:
            return False, f"frame {i} shape mismatch {a.shape} vs {b.shape}"
        chunks_a.append(a)
        chunks_b.append(b)
    aa, bb = np.stack(chunks_a), np.stack(chunks_b)
    if size is not None:
        aa, bb = aa * 1000.0, bb * 1000.0
    ok, msg = diff_msg(aa, bb, lossy)
    return ok, f"{n_orig}f, {len(idxs)} frames checked, {msg} (mm)"


def decode_all_gray12(path, h=H, w=W, pix="gray12le"):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", pix, "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<u2").reshape(-1, h, w)


def decode_all_rg8(path, h=H, w=W, pack="8+4"):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    return unpack_rg8(rgb[..., 0], rgb[..., 1], pack), rgb


def check_full(orig, recon, size=None, src_rg8=False, pack="8+4", src_pack="8+4",
               kind="rg8", bits=12, lossy=False):
    """Compare every pixel of every frame. Returns (ok, message)."""
    if src_rg8:
        o, o_rgb = decode_all_rg8(orig, pack=src_pack)
        if o_rgb[..., 2].any():
            return False, "source blue channel not zero"
    else:
        o = decode_all_gray12(orig)
    o = clip_src(o, bits)
    if kind == "gray":
        pix = f"gray{bits}le"
        if size is None:
            r = decode_all_gray12(recon, pix=pix)
        else:
            r = decode_all_gray12(recon, size, size, pix=pix)
    else:
        if size is None:
            r, rgb = decode_all_rg8(recon, pack=pack)
        else:
            r, rgb = decode_all_rg8(recon, size, size, pack=pack)
        if rgb[..., 2].any():
            return False, "blue channel not zero"
    if size is not None:
        o = np.stack([dataloader_depth_m(f, size) for f in o]) * 1000.0
        r = np.stack([dataloader_depth_m(f, size) for f in r]) * 1000.0
    if o.shape != r.shape:
        return False, f"shape mismatch orig={o.shape} recon={r.shape}"
    ok, msg = diff_msg(o, r, lossy)
    return ok, f"{o.shape[0]}f x {o.shape[1]}x{o.shape[2]} = {o.size} px, {msg} (mm)"


def check_repaired(orig, recon):
    """Check a file rebuilt by repair_corrupt_to_rg8.py from a damaged source.

    The contract there is weaker than bit-exact-everywhere: frame count must match
    the source, every frame still recoverable from the source must be bit-exact, and
    every unrecoverable position must be exactly 0 ("no depth").
    """
    from repair_corrupt_to_rg8 import decode_recoverable

    n_orig, n_recon = nb_frames(orig), nb_frames(recon)
    if n_orig != n_recon:
        return False, f"frame count mismatch orig={n_orig} recon={n_recon}"

    frames, recovered = decode_recoverable(orig, n_orig)
    r, b_max = load_rg8_frames(recon, list(range(n_orig)), pack="8+4")
    if b_max != 0:
        return False, f"blue channel not zero (max={b_max})"

    for i in range(n_orig):
        got = r[i]
        if i in recovered:
            if not np.array_equal(got, frames[i]):
                worst = int(np.abs(got.astype(np.int32) - frames[i].astype(np.int32)).max())
                return False, f"recovered frame {i} differs (max_abs_diff={worst})"
        elif got.any():
            return False, f"unrecoverable frame {i} is not zero-filled"
    return True, (f"{n_orig}f, {len(recovered)} recovered exact, "
                  f"{n_orig - len(recovered)} zero-filled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20,
                    help="number of random files to spot-check")
    ap.add_argument("--frames", type=int, default=5,
                    help="frames per sampled file (evenly spaced)")
    ap.add_argument("--full", type=int, default=2,
                    help="number of files to compare in full (every frame, every pixel)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=ROOT, help="videos root holding both directories")
    ap.add_argument("--src-suffix", default=SRC_SUFFIX,
                    help="suffix of the uint12 original directories")
    ap.add_argument("--dst-suffix", default="",
                    help="suffix of the dest directories (e.g. _rg8_224, _rg8_55_224, _g10_224, _g10_224_crf035)")
    ap.add_argument("--pack", default=None, choices=["8+4", "5+5"],
                    help="RG8 unpack. Default: infer from --dst-suffix (_rg8_55* -> 5+5)")
    ap.add_argument("--src-rg8", action="store_true",
                    help="the source side is itself RG8, not gray12le: use to check a "
                         "downscaled set against the full-resolution RG8 it came from")
    ap.add_argument("--out-size", default="native",
                    help="'native' to compare millimetres at equal resolution, or the "
                         "square size of a downscaled RG8 set (e.g. 224) to compare the "
                         "depth_m arrays the dataloader produces from either side")
    ap.add_argument("--repaired-list", default=None,
                    help="file of relative <cam>/<chunk>/<file>.mp4 paths rebuilt from "
                         "damaged sources; checked separately and excluded from sampling")
    ap.add_argument("--interactive", action="store_true",
                    help="breakpoint() on the first sampled file with orig0/recon0 in scope")
    args = ap.parse_args()

    root = args.root
    size = None if args.out_size == "native" else int(args.out_size)
    pack = args.pack or infer_pack(args.dst_suffix)
    src_pack = infer_pack(args.src_suffix) if args.src_rg8 else "8+4"
    kind = infer_kind(args.dst_suffix)
    bits = infer_bits(args.dst_suffix)
    lossy = infer_lossy(args.dst_suffix)

    repaired = []
    if args.repaired_list:
        with open(args.repaired_list) as f:
            repaired = [ln.strip() for ln in f if ln.strip()]

    pairs = find_pairs(root, args.src_suffix, args.dst_suffix)
    print(f"[scan] {len(pairs)} re-encoded files with an original available "
          f"(<cam>{args.src_suffix} [{'rg8' if args.src_rg8 else 'gray12le'}] vs "
          f"<cam>{args.dst_suffix}, kind={kind} bits={bits} lossy={lossy} pack={pack}, "
          f"compare={'mm' if size is None else f'depth_m @ {size}'})")
    if repaired:
        skip = {os.path.join(root, rel) for rel in repaired}
        pairs = [(o, r) for o, r in pairs if r not in skip]
        print(f"[scan] {len(repaired)} repaired files checked separately, "
              f"{len(pairs)} eligible for sampling")
    if not pairs:
        print("[scan] nothing to verify.")
        return 1

    rng = random.Random(args.seed)
    sample = rng.sample(pairs, min(args.samples, len(pairs)))

    if args.interactive:
        orig, recon = sample[0]
        print(f"[interactive] orig  = {orig}")
        print(f"[interactive] recon = {recon}")
        orig0 = (load_rg8_frames(orig, [0], src_pack)[0] if args.src_rg8
                 else load_gray12_frames(orig, [0]))[0]
        recon0 = load_recon_frames(recon, [0], kind, pack)[0][0]
        print(f"[interactive] orig0  shape={orig0.shape} dtype={orig0.dtype} "
              f"min={orig0.min()} max={orig0.max()}")
        print(f"[interactive] recon0 shape={recon0.shape} dtype={recon0.dtype} "
              f"min={recon0.min()} max={recon0.max()}")
        # Try: (orig0 == recon0).all() / orig0.tobytes() == recon0.tobytes()
        breakpoint()

    failures = []
    for i, (orig, recon) in enumerate(sample, 1):
        try:
            ok, msg = check_sampled(orig, recon, args.frames, size, args.src_rg8, pack, src_pack,
                                   kind, bits, lossy)
        except Exception as e:  # noqa
            ok, msg = False, repr(e)
        tag = "PASS" if ok else "FAIL"
        print(f"[sampled {i}/{len(sample)}] {tag} {os.path.relpath(recon, root)}  {msg}",
              flush=True)
        if not ok:
            failures.append((recon, msg))

    full = sample[:args.full] if args.full else []
    for i, (orig, recon) in enumerate(full, 1):
        try:
            ok, msg = check_full(orig, recon, size, args.src_rg8, pack, src_pack,
                                kind, bits, lossy)
        except Exception as e:  # noqa
            ok, msg = False, repr(e)
        tag = "PASS" if ok else "FAIL"
        print(f"[full {i}/{len(full)}] {tag} {os.path.relpath(recon, root)}  {msg}",
              flush=True)
        if not ok:
            failures.append((recon, msg))

    for i, rel in enumerate(repaired, 1):
        cam, chunk, fn = rel.split("/")
        recon = os.path.join(root, cam, chunk, fn)
        orig = os.path.join(root, cam + args.src_suffix, chunk, fn)
        try:
            ok, msg = check_repaired(orig, recon)
        except Exception as e:  # noqa
            ok, msg = False, repr(e)
        tag = "PASS" if ok else "FAIL"
        print(f"[repaired {i}/{len(repaired)}] {tag} {rel}  {msg}", flush=True)
        if not ok:
            failures.append((recon, msg))

    if failures:
        print(f"\n[done] {len(failures)} FAILURES:")
        for path, msg in failures:
            print(f"  {path}\t{msg}")
        return 1
    print(f"\n[done] all checks passed "
          f"({len(sample)} sampled x {args.frames} frames, {len(full)} full, "
          f"{len(repaired)} repaired)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
