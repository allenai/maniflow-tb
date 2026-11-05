# RGB 224 videos

`reencode_to_224_batch.py` creates 224×224 sidecar videos using the same
Pillow bilinear resize as the dataloader. The resize is checked on the first
frame before encoding. Outputs use lossy AV1 by default; use
`--codec lossless` when the encoded file itself must be bit-exact.

## Batch re-encode

`reencode_to_224_batch.py` reads each `observation.images.*` camera under
`--src-root` (dataset root or `videos/`) and writes
`<camera>_224/chunk-XXX/file-YYY.mp4` next to the originals.

```bash
python3 scripts/rgb_conversion/reencode_to_224_batch.py \
    --src-root /path/to/dataset \
    --workers 32 --limit 4
```

Default is front+gripper if those dirs exist, otherwise every
`observation.images.*` that is not already a `_<size>` output. Override with
`--cams`. Writes are atomic and existing outputs are skipped.

```bash
python3 scripts/rgb_conversion/reencode_to_224_batch.py \
    --src-root /path/to/dataset \
    --workers 32 --codec av1 --crf 26 --preset 8
```

| Flag | Default | Notes |
|------|---------|--------|
| `--codec av1` | yes | `libsvtav1`, `yuv420p` |
| `--codec lossless` | | `libx264rgb -qp 0 -g 1` (bit-exact, ~11 TB FG on the reorient set) |
| `--crf` | **26** | SVT-AV1 quality (training default). Not x264's CRF 23 — that is ~AV1 30. Ignored for lossless. |
| `--preset` | `8` / `medium` | SVT-AV1 **speed** 0–13 (higher=faster, a bit less efficient at the same CRF). `8` is the batch default. x264 uses named presets (`medium`). |
| `--size` | 224 | must match training `img_size` |
| `--limit N` | 0 | first N files; use for a pilot |

## Verify before swapping anything

```bash
python3 scripts/rgb_conversion/verify_reencode.py \
    --root /path/to/dataset --samples 20 --frames 8 --full 2
```

Reports `max_abs` / PSNR of `decode(dst)` vs `PIL_bilinear(decode(src))`.
For lossless output add `--require-lossless` (fails unless `max_abs=0`).

## What the dataloader should read

RGB originals stay untouched. Converted videos live in
`<camera>_224` beside them. Training defaults to 640×360. Pass `--rgb_224`
(`datasets.vla_data.use_rgb_224 true`) to read the sidecars and skip the
dataloader resize. Missing `_224` videos fail the run.

Leave `meta/info.json` at 640×360; heatmap / `target_xy` still use native
resolution.

## Gotchas

- **Resize is exact; AV1 is not.** The contract is bit-exact *before*
  encode. Lossy 224 AV1 will not reproduce the uint8 the model sees today.
- **Lossless is large.** Front+gripper FFV1 / x264rgb on the reorient set
  is ~11 TB, not the ~500 GB you get from lossy AV1 scaled by pixel count.
- **Convert every root you train on.** Mixing 640 and 224 keys in one
  `--data_root_dir` is undefined until the dataloader knows about `_224`.
- **One file per episode, not one image per frame.** Packed mp4s keep Weka
  inode counts in the same ballpark as today.
