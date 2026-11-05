"""Bit-exact RGB resize used by the training dataloader.

``hvla.dataloader.franka_dataset._pil_to_tensor`` does:

    pil = Image.fromarray(img)                      # uint8 HWC RGB
    pil = pil.resize((size, size), Image.BILINEAR)
    return to_tensor(pil)                           # float32 / 255

This module is the first two lines. ``to_tensor`` on uint8 RGB is exactly
``/ 255``, so the uint8 we return is the array training would convert.

Do **not** replace this with ffmpeg ``scale``, OpenCV ``INTER_LINEAR``, or
torchvision ``BILINEAR``. Those kernels and rounding rules are different —
on these 640x360 clips ffmpeg ``flags=bilinear`` disagrees with Pillow on
~70% of 224x224 pixels (max abs 35). The only way to stay bit-exact with
the dataloader is to call the same Pillow API on the same uint8 RGB frame.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Same enum the dataloader passes. Image.Resampling.BILINEAR is identical
# (both are 2) but keep the attribute name in lockstep with _pil_to_tensor.
_DATALOADER_RESAMPLE = Image.BILINEAR


def resize_rgb_uint8(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize one uint8 HWC RGB frame the way ``_pil_to_tensor`` does.

    Returns uint8 HWC of shape ``(size, size, 3)``.
    """
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB, got shape={img.shape}")
    if img.dtype != np.uint8:
        raise TypeError(f"expected uint8, got {img.dtype}")
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), _DATALOADER_RESAMPLE)
    return np.asarray(pil, dtype=np.uint8)


def dataloader_pil_bilinear_ref(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Independent copy of the dataloader resize, for before-encode asserts.

    Written as a second, literal transcription so a future edit to
    ``resize_rgb_uint8`` cannot silently make the self-check tautological.
    """
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)
