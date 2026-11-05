# RG8 Depth Videos

Depth is stored as uint12 grayscale (`gray12le`), which needs slow PyAV keyframe-seeking.
**RG8** packs the same value into an RGB video — `R = depth & 0xFF`, `G = depth >> 8`,
`depth = R + G * 256` — losslessly (`libx264rgb -qp 0 -g 1`), decoded via torchcodec.

**No dataset ships with RG8 depth**; nothing in the merge/datagen pipeline produces it, so
you must convert each root yourself.

## 1. Create the RG8 videos

`reencode_to_rg8_batch.py` reads cameras from `--src-root` and writes `<camera>_rg8` beside them.

```python
W, H = 640, 360            # native depth resolution, NOT 224; wrong values fail every file
CAMS = ["observation.depth.camera:front_camera", "observation.depth.camera:gripper_camera"]
```

```bash
python3 scripts/depth_conversion/reencode_to_rg8_batch.py \
    --src-root /path/to/dataset/videos \
    --workers 96 --fail-log /path/you/can/write/failures.log
```

Writes are atomic and existing outputs are skipped (resumable). Check `[done] ok=… fail=…`.
`--fail-log` defaults to alongside the source root.

### `--pack 5+5`: 10-bit high/low split

Clips to 1023 mm: `R = mm >> 5`, `G = mm & 31`, `depth = (R << 5) | G`.
Writes `<camera>_rg8_55` or `<camera>_rg8_55_224`. ~2× smaller than 8+4 at GOP 1.

```bash
python3 scripts/depth_conversion/reencode_to_rg8_batch.py \
    --src-root /path/to/dataset/videos --out-size 224 --pack 5+5
python3 scripts/depth_conversion/verify_reencode.py \
    --root /path/to/dataset/videos --src-suffix "" --dst-suffix _rg8_55_224 --out-size 224
```

Train with `--depth_rg8 rg8_55_224`.

### `--out-size 224`: pre-downscaled RG8

Nearest-downscales to 224×224 before the split (`<camera>_rg8_224`): ~0.34× bytes, ~3× faster
decode, same `depth_m` the model sees. Do **not** reimplement that resize with `floor()` —
Pillow's NEAREST mapping differs for float32 vs uint16 and shifts 20 of 224 columns
(2.5% of pixels). The script reads the mapping out of Pillow; see its docstring.
`_decode_depth_raw` skips resize when the frame is already `img_size` (train at 224).

## 2. Verify before swapping anything

`verify_reencode.py` checks a converted root against the uint12 originals with the
dataloader decoders. For `--out-size 224` the resolutions differ, so pass the same size
and it compares the final `depth_m` arrays:

```bash
python3 scripts/depth_conversion/verify_reencode.py \
    --root /path/to/dataset/videos --src-suffix "" --dst-suffix _rg8_224 \
    --out-size 224 --samples 60 --frames 8 --full 6
```

## 3. Where the dataloader reads RG8 from

`depth_rg8` appends the variant to the depth key and reads
`<data_root_dir>/videos/<depth_key>_<variant>/chunk-NNN/file-NNN.mp4`, so converted
videos stay beside the uint12 originals and the base dirs keep full-precision depth.
A missing variant path does not misread gray12 as RGB — it silently trains on zeros
(see gotchas). Leave `meta/info.json` as `gray12le_mm`; `_decode_depth_raw` picks the
decoder from config, not that tag, because a re-encode inherits it. `ffprobe` a
converted file to confirm `pix_fmt=rgb24`. RGB, `data/`, and `meta/` are untouched.

## 4. Train with `depth_rg8`

Variant must match the directory suffix: `rg8` / `rg8_224` (8+4), `rg8_55` / `rg8_55_224` (5+5).

```bash
python scripts/submit_maniflow_train.py ... --depth_rg8            # bare flag means rg8
python scripts/submit_maniflow_train.py ... --depth_rg8 rg8_224    # run name gets an _rg8_224 suffix
--datasets.vla_data.depth_rg8 rg8_224                              # raw train_hvla.py override
depth_rg8=rg8_224                                                  # set in scripts/debug_maniflow_train.sh
```

Default is `null` (PyAV uint12 on the base depth videos) on train and val.
Legacy `true`/`false` still parse as `rg8` and `null`.

## Gotchas

- **`depth_frames/*.npy` silently wins.** `__getitem__` tries npy first, so `depth_rg8` is a
  no-op on roots merged with `--depth_frames`. RG8 only helps roots without that tree.
- **An unconverted variant trains on zeros.** A missing `<depth_key>_<variant>` dir is omitted
  from `depth_maps` and collate fills zeros. `ls` the suffixed dir first. Invalid names
  (not `rg8` plus `_` tokens) raise in `normalize_depth_rg8`.
- **Convert every root.** `depth_rg8` is global; mixed roots zero the unconverted half.
- **Only `gray12le_mm` datasets are affected.** `dual_row_uint16_mm` and `gray12le_mm_pyav`
  ignore the flag. Dispatch is in `hvla/dataloader/franka_dataset.py` (`_decode_depth_raw`).

# g12 / g10 Depth Videos (HEVC gray, short GOP)

RG8 is exact but every frame is intra: ~16 / 10 KB/frame (front / gripper) at 224×224, ~2.2 TB
on 37k episodes — too big for the page cache, and a cold RG8 file costs torchcodec 0.1–6 s
(mp4 index at EOF). **g12** stores the same 12-bit depth as lossless gray with temporal prediction:

    libx265 lossless=1 bframes=0 keyint=G open-gop=0 scenecut=0, gray12le, 224x224, +faststart
    G = 8 front, 4 gripper

Measured on a 224x224 depth dataset (random access, one process):

| variant | KB/frame front / gripper | warm ms/frame | cold-file ms/frame |
|---|---|---|---|
| original gray12le 640×360, 1 keyframe/episode (`depth_rg8: null`) | 6.4 / 11.5 | 490 / 740 | 900-2600 |
| `rg8_224` (torchcodec) | 16.0 / 10.4 | 4.0 / 3.3 | 100-6000 |
| `g12_224` (PyAV) | 3.3 / 5.2 | 2.8 / 2.8 | 3-80 |

`bframes=0` makes decode order = display order: seek to the prior keyframe, decode at most G
frames. Bit-exact for lossless.

## 1. Convert

`reencode_to_g12_batch.py` transcodes the `_rg8_224` videos (the exact 224×224 values
training reads) into `<depth_key>_g12_224`, one mp4 per episode, same chunk/file layout.
Writes are atomic and existing outputs are skipped, so it is resumable; `--shard I
--num-shards N` splits the sorted file list for parallel CPU-only jobs (~10 ms/frame/core
front, ~20 ms gripper with `--preset medium`). `meta/info.json` is not touched; a sidecar
`meta/depth_g12_224.json` records codec, GOPs and preset.

```bash
python3 scripts/depth_conversion/reencode_to_g12_batch.py \
    --src-root /path/to/dataset --workers 32 \
    --gop 8 --gop-cam observation.depth.camera:gripper_camera=4 \
    --shard 0 --num-shards 16          # --dst-root and --limit for pilots
```

The source must be RG8 (`h264`/`gbrp`) and depth must fit 12 bits (max 4095 mm); the
tool asserts both and checks the output frame count. `ffprobe` a converted file:
`hevc` / `gray12le`.

## 1b. g10 (10-bit gray HEVC)

`--bits 10` clips to 1023 mm and writes gray10le. Omit `--crf` for lossless; `--crf 0.35` is lossy.

```bash
python3 scripts/depth_conversion/reencode_to_g12_batch.py \
    --src-root /path/to/dataset --src-suffix "" --bits 10 --out-size 224 \
    --gop 8 --gop-cam observation.depth.camera:gripper_camera=4
python3 scripts/depth_conversion/reencode_to_g12_batch.py \
    --src-root /path/to/dataset --src-suffix "" --bits 10 --out-size 224 --crf 0.35 \
    --gop 8 --gop-cam observation.depth.camera:gripper_camera=4
python3 scripts/depth_conversion/verify_reencode.py \
    --root /path/to/dataset/videos --src-suffix "" --dst-suffix _g10_224 --out-size 224
python3 scripts/depth_conversion/verify_reencode.py \
    --root /path/to/dataset/videos --src-suffix "" --dst-suffix _g10_224_crf035 --out-size 224
```

Train with `--depth_g12 g10_224` or `--depth_g12 g10_224_crf035`.

## 2. Train with `depth_g12`

```bash
python scripts/submit_maniflow_train.py ... --depth_g12 g12_224     # run name gets a _g12_224 suffix
--datasets.vla_data.depth_g12 g12_224
```

`depth_g12` names the directory suffix, reaches train and val, and is mutually exclusive
with `depth_rg8`. A missing `<depth_key>_<variant>` dir **raises** at init instead of
training on zeros. The decoder (`_decode_g12_frame_pyav`) needs PyAV; any version works
because it reads gray12le from the frame plane, not through `to_ndarray()`.

## Gotchas

- **Frame choice must match torchcodec.** Parquet `timestamp` is float32 and sits up to 4e-6 s
  off `k / fps`; torchcodec (`seek_mode="approximate"`) returns `floor(ts * fps)`, so about
  half of reads are frame `k-1`. `_decode_g12_frame_pyav` uses the same rule. The old PyAV
  path picks the closest frame. Do not fix this in one decoder only.
- **`depth_frames/*.npy` silently wins** over g12/g10 too (see RG8 gotcha).
- **Convert every root** passed to `--data_root_dir`; the flag is global.
