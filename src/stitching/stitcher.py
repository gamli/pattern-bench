"""Functions for registration and composition of full-resolution scan arrays.

Pipeline:
    SIFT feature detection and description -> pairwise affine matching with
    RANSAC -> robust global affine registration -> one warp per original ->
    arithmetic mean of all overlapping image contributions.

The model is full affine (six degrees of freedom), including shear. The first
image fixes the reference coordinates and pixel scale. Composition preserves
this orientation and only translates the common bounds onto the output canvas.
Overlaps are averaged to expose registration errors; seam finding is deferred.

Usage (images are uint8 OpenCV BGR arrays):
    configure_opencv() # Optional, explicit application/notebook setup.
    result = stitch_images(test_images)
    stitched_image = result.composition.image
    print(result.registration.rms_error)

Individual stages:
    features = detect_features(test_images)
    matches = match_features(features)
    transforms = estimate_global_affine(matches, len(test_images))
    composition = compose_images(test_images, transforms)

The processing functions do not read/write files or configure OpenCV execution
settings. Input arrays are not modified or resized. Diagnostics are returned
as point correspondences, residuals, transforms, and per-pixel coverage counts.

Tested with Python 3.11+, NumPy 2.3.5, OpenCV 4.13.0 (wheel 4.13.0.92).
No bitwise reproducibility guarantee: the observed differences between cached
and freshly computed SIFT features still need investigation (see the TODOs).
"""

from dataclasses import dataclass

from src.stitching.composition import CompositionResult, compose_images
from src.stitching.registration import RegistrationResult, register_images
from src.stitching.shared import StitchingConfig
from stitching.shared import StitchingImages


@dataclass(frozen=True)
class StitchingResult:
    registration: RegistrationResult
    composition: CompositionResult


def stitch_images(
        stitching_image: StitchingImages,
        config: StitchingConfig,
) -> StitchingResult:
    """Run Registration and Composition; input image arrays are not modified."""
    registration = register_images(stitching_image, config)
    composition = compose_images(registration.transforms, stitching_image, config)
    return StitchingResult(registration, composition)
