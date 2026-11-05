"""Depth augmentation for sim-to-real transfer on raw decoded depth."""

import cv2
import numpy as np

DEFAULT_BASELINE_M = 0.120  # 120 mm as in ZED2

DEFAULT_DEPTH_NOISE_KWARGS = {
    "sigma_color_rgb": 10.0,
    "sigma_space": 3.5,
    "guide_bits": 2,
    "diameter": 21,
    "ideal_mix": 0.6,
}


def gate(disp: np.ndarray) -> np.ndarray:
    """Clamp disparity to avoid division by zero when converting back to depth."""
    return np.maximum(disp, 0.01)


def quantize_guide_gray(gray: np.ndarray, guide_bits: int = 8) -> np.ndarray:
    """Reduce guide to guide_bits significant bits (1-8). 8 = no-op."""
    guide_bits = int(np.clip(guide_bits, 1, 8))
    if guide_bits >= 8:
        return gray
    shift = 8 - guide_bits
    return ((gray.astype(np.uint16) >> shift) << shift).astype(np.uint8)


def add_rgb_guided_drift(
    disp: np.ndarray,
    gray: np.ndarray,
    sigma_space: float,
    sigma_color_rgb: float,
    diameter: int = 7,
    guide_bits: int = 8,
) -> np.ndarray:
    """RGB-guided joint bilateral filter on disparity (photometric range kernel)."""
    if gray.shape != disp.shape:
        raise ValueError(f"gray shape {gray.shape} must match disp shape {disp.shape}")
    if gray.dtype != np.uint8:
        raise ValueError(f"gray must be uint8, got {gray.dtype}")

    valid = disp > 0
    # TODO: assumes fully valid depth — OpenCV jointBilateralFilter has no validity mask,
    # so zero/invalid disparity still participates as neighbors during filtering.
    gray_q = quantize_guide_gray(gray, guide_bits)
    gray_f = gray_q.astype(np.float32)
    disp_f = disp.astype(np.float32, copy=False)
    filtered = cv2.ximgproc.jointBilateralFilter(
        gray_f, disp_f, diameter, sigma_color_rgb, sigma_space,
    )
    out = gate(filtered.astype(np.float32, copy=False))
    out[~valid] = 0.0
    return out


def blend_ideal_disparity(
    clean_disp: np.ndarray,
    corrupted_disp: np.ndarray,
    ideal_mix: float,
) -> np.ndarray:
    """Blend clean and corrupted disparity: ideal_mix * clean + (1 - ideal_mix) * corrupted."""
    out = np.zeros_like(clean_disp, dtype=np.float32)
    valid = (clean_disp > 0) & (corrupted_disp > 0)
    out[valid] = clean_disp[valid] * ideal_mix + corrupted_disp[valid] * (1.0 - ideal_mix)
    return out


def add_noise_to_raw_depth(
    depth_m: np.ndarray,
    fx: float,
    baseline_m: float = DEFAULT_BASELINE_M,
    *,
    gray: np.ndarray | None = None,
    sigma_space: float = DEFAULT_DEPTH_NOISE_KWARGS["sigma_space"],
    sigma_color_rgb: float = DEFAULT_DEPTH_NOISE_KWARGS["sigma_color_rgb"],
    guide_bits: int = DEFAULT_DEPTH_NOISE_KWARGS["guide_bits"],
    diameter: int = DEFAULT_DEPTH_NOISE_KWARGS["diameter"],
    ideal_mix: float = DEFAULT_DEPTH_NOISE_KWARGS["ideal_mix"],
    max_depth_m: float | None = 2.5,
) -> np.ndarray:
    """Augment clean simulated depth in meters (H, W) float32.

    Disparity -> RGB-guided joint bilateral -> blend in disparity space -> depth.
    ideal_mix * clean_disp + (1 - ideal_mix) * corrupted_disp, then fb / disp.
    Requires gray guide image. If gray is None, returns clean depth unchanged.
    Invalid pixels (depth <= 0) stay zero throughout.
    """
    depth_m = depth_m.astype(np.float32, copy=False)
    valid = depth_m > 0
    if not np.any(valid) or gray is None:
        return depth_m

    fb = fx * baseline_m
    clean_disp = np.zeros_like(depth_m, dtype=np.float32)
    clean_disp[valid] = fb / depth_m[valid]

    corrupted_disp = add_rgb_guided_drift(
        clean_disp.copy(), gray, sigma_space, sigma_color_rgb,
        diameter=diameter, guide_bits=guide_bits,
    )

    if ideal_mix >= 1.0:
        return depth_m
    if ideal_mix <= 0.0:
        blended_disp = corrupted_disp
    else:
        blended_disp = blend_ideal_disparity(clean_disp, corrupted_disp, ideal_mix)

    out = np.zeros_like(depth_m, dtype=np.float32)
    out[valid] = fb / gate(blended_disp[valid])
    out[~valid] = 0.0
    if max_depth_m is not None:
        out[out > max_depth_m] = 0.0
    return out
