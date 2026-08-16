import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .estimate_global_affine import estimate_global_affine
from .feature_detection import detect_features
from .feature_matching import PairwiseMatch, match_features
from .shared import FloatArray, StitchingConfig, StitchingImages

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistrationResult:
    """Transforms map each source image to the first image's coordinate frame.

    transforms[0] is identity. residuals are Euclidean distances between
    corresponding points after both have been transformed to this frame.
    These residuals do not measure the absolute physical accuracy of the scans.
    """

    transforms: tuple[FloatArray, ...]
    pairwise_matches: tuple[PairwiseMatch, ...]
    feature_counts: tuple[int, ...]
    residuals: FloatArray

    @property
    def rms_error(self) -> float | None:
        if not self.residuals.size:
            return None
        return float(np.sqrt(np.mean(self.residuals ** 2)))


def register_images(
        stitching_images: StitchingImages,
        config: StitchingConfig,
) -> RegistrationResult:
    """Feature Detection -> Feature Matching -> Global Affine Registration."""

    if stitching_images.count == 1:
        return RegistrationResult((np.eye(3),), (), (0,), np.empty(0, dtype=np.float64))

    features = detect_features(stitching_images, config)

    matches = match_features(features, stitching_images, config)

    transforms = estimate_global_affine(matches, stitching_images, config)

    result = RegistrationResult(
        transforms=transforms,
        pairwise_matches=matches,
        feature_counts=tuple(len(feature.getKeypoints()) for feature in features),
        residuals=registration_residuals(transforms, matches),
    )

    _LOGGER.info("Registration: all %d images connected; RMS %.6f original pixels", stitching_images.count,
                 result.rms_error)

    return result


def registration_residuals(
        transforms: Sequence[FloatArray],
        matches: Sequence[PairwiseMatch],
) -> FloatArray:
    """Distances between matched points in the common reference coordinate frame."""
    errors = []
    for pair in matches:
        if not pair.accepted:
            continue
        gi, gj = transforms[pair.image_idx_i], transforms[pair.image_idx_j]
        common_i = pair.keypoints_i @ gi[:2, :2].T + gi[:2, 2]
        common_j = pair.keypoints_j @ gj[:2, :2].T + gj[:2, 2]
        errors.append(np.linalg.norm(common_i - common_j, axis=1))
    return np.concatenate(errors) if errors else np.empty(0, dtype=np.float64)
