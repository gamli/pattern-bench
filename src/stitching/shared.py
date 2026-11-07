from dataclasses import dataclass
from functools import cached_property
from math import isfinite
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from scanner.constants import SCANNER_DPI
from utils.md_logger import MdLogger

ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StitchingImages:
    scan_files: tuple[Path, ...]
    scan_images: tuple[ImageArray, ...]
    scan_dpis: tuple[SCANNER_DPI | None, ...]

    def __post_init__(self) -> None:
        if self.count == 0:
            raise ValueError("Provide at least one image.")
        for file, image in self.files_and_images:
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                raise ValueError(f"Image {file} must be a uint8 NumPy array.")
            if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
                raise ValueError(f"Image {file} must have shape (height, width, 3) in BGR order.")

    @property
    def count(self) -> int:
        return len(self.scan_files)

    @cached_property
    def files_and_images(self) -> tuple[tuple[Path, ImageArray], ...]:
        return tuple(zip(self.scan_files, self.scan_images))

    @cached_property
    def files_and_images_and_dpi(self) -> tuple[tuple[Path, ImageArray, SCANNER_DPI], ...]:
        return tuple(zip(self.scan_files, self.scan_images, self.scan_dpis))


@dataclass(frozen=True)
class StitchingConfig:
    """Parameters used in the original-resolution analysis.

    OpenCV's match_conf is not the conventional Lowe ratio: its CPU matcher
    accepts d_best < (1 - match_conf) * d_second, in both matching directions.
    Thus, 0.65 corresponds to a descriptor distance ratio below 0.35.
    The native affine matcher's RANSAC threshold is its default: 3 pixels.
    huber_delta is measured in the reference image's original pixel units.
    pixel_refinement enables pixel pair validation and joint intensity fitting.
    pixel_min_correlation measures agreement of local image contrast; uniform
    regions of any colour provide no matching evidence. pixel_max_correction
    bounds actual scan-corner movement in original pixels, including rotation.
    pixel_max_samples caps samples per overlap and pyramid level; the final
    level uses original-resolution pixels.
    """

    match_conf: float = 0.65
    num_matches_thresh1: int = 6
    confidence_threshold: float = 1.0
    min_inliers: int = 20
    min_scale: float = 0.2
    max_scale: float = 5.0
    max_anisotropy: float = 5.0
    huber_delta: float = 1.0
    irls_iterations: int = 5

    vlog: MdLogger | None = None
    use_fixed_seed: bool = False
    max_threads: int = 1
    use_gpu: bool = False

    pixel_refinement: bool = True
    pixel_min_correlation: float = 0.5
    pixel_max_correction: float = 20.0
    pixel_refinement_iterations: int = 50
    pixel_max_samples: int = 20_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.match_conf < 1.0:
            raise ValueError("match_conf must be in [0, 1).")
        if self.num_matches_thresh1 < 3 or self.min_inliers < 3:
            raise ValueError("An affine estimate needs at least three point pairs.")
        if not 0.0 < self.min_scale < self.max_scale or self.max_anisotropy < 1.0:
            raise ValueError("Invalid affine scale limits.")
        if self.confidence_threshold < 0 or self.huber_delta <= 0:
            raise ValueError("Invalid confidence threshold or Huber delta.")
        if self.irls_iterations < 0:
            raise ValueError("irls_iterations must be >= 0.")
        if not 0.0 < self.pixel_min_correlation <= 1.0:
            raise ValueError("pixel_min_correlation must be in (0, 1].")
        if not isfinite(self.pixel_max_correction) or self.pixel_max_correction <= 0:
            raise ValueError("pixel_max_correction must be finite and positive.")
        if self.pixel_refinement_iterations < 1 or self.pixel_max_samples < 128:
            raise ValueError("Pixel refinement needs at least one iteration and 128 samples per pair.")

    def configure_opencv(self) -> None:
        cv.setNumThreads(self.max_threads)
        cv.ocl.setUseOpenCL(self.use_gpu)

    def apply_fixed_seed(self) -> None:
        if self.use_fixed_seed:
            cv.setRNGSeed(0x5A17C0DE)
