import itertools
from dataclasses import dataclass
from typing import Sequence

import cv2 as cv
import numpy as np

from cli.commands.utils import print_info, start_info_timer
from .shared import FloatArray, StitchingConfig, StitchingImages


@dataclass(frozen=True)
class PairwiseMatch:
    """
    One unordered image pair; points use original image coordinates.

    affine_i_to_j is OpenCV's pairwise estimate, not a global transform.
    points_i and points_j contain only RANSAC inliers of accepted pairs.
    Rejected pairs remain available for diagnostics, with empty point arrays.
    """

    image_idx_i: int
    image_idx_j: int
    num_matches: int
    num_inliers: int
    confidence: float
    affine_ij: FloatArray | None
    keypoints_i: FloatArray
    keypoints_j: FloatArray
    rejection_reason: str | None

    @property
    def accepted(self) -> bool:
        return self.rejection_reason is None


def match_features(
        image_features: Sequence[cv.detail.ImageFeatures],
        stitching_images: StitchingImages,
        config: StitchingConfig,
) -> tuple[PairwiseMatch, ...]:
    """Match every image pair using the native full-affine OpenCV matcher.

    The matcher performs bidirectional descriptor matching and affine RANSAC.
    Its acceptance filters are exactly those of the full-resolution analysis.
    No spatial ordering or adjacency is inferred from image names or indices.

    Execution settings and RNG initialization belong to the caller. This
    function does not set a seed, thread count, or OpenCL policy.
    """

    print_info("Matching Features")
    info_timer = start_info_timer()

    image_count = stitching_images.count

    if image_count == 1:
        return ()

    matcher = cv.detail.AffineBestOf2NearestMatcher(
        full_affine=True,
        try_use_gpu=config.use_gpu,
        match_conf=config.match_conf,
        num_matches_thresh1=config.num_matches_thresh1,
    )

    config.apply_fixed_seed()

    print_info("Apply OpenCV Matcher")
    all_pair_matches_infos = matcher.apply2(image_features)
    info_timer.print("\tDone")

    # just to release buffers and stuff early to save a little on memory
    matcher.collectGarbage()

    keypoints = [feature.getKeypoints() for feature in image_features]

    pairs = []

    print_info("Handle Match results for all image pairs")

    for i, j in itertools.combinations(range(image_count), 2):

        matches_info = all_pair_matches_infos[i * image_count + j]

        matches = matches_info.getMatches()

        num_matches = len(matches)

        confidence = float(matches_info.confidence)

        num_inliers = int(matches_info.num_inliers)

        affine_ij = (None if matches_info.H is None
                     else np.asarray(matches_info.H, dtype=np.float64).copy())

        rejection_reason = _rejection_reason(
            affine_ij=affine_ij,
            confidence=confidence,
            num_inliers=num_inliers,
            config=config,
        )

        if rejection_reason is None:
            inliers = np.asarray(matches_info.getInliers(), dtype=bool).reshape(-1)
            keypoints_i = np.array([keypoints[i][m.queryIdx].pt for m in matches])[inliers]
            keypoints_j = np.array([keypoints[j][m.trainIdx].pt for m in matches])[inliers]
        else:
            keypoints_i = np.empty((0, 2), dtype=np.float64)
            keypoints_j = np.empty((0, 2), dtype=np.float64)

        pair = PairwiseMatch(
            image_idx_i=i,
            image_idx_j=j,
            num_matches=num_matches,
            num_inliers=num_inliers,
            confidence=confidence,
            affine_ij=affine_ij,
            keypoints_i=keypoints_i,
            keypoints_j=keypoints_j,
            rejection_reason=rejection_reason,
        )
        pairs.append(pair)

    info_timer.print("\tDone")

    return tuple(pairs)


def _rejection_reason(
        affine_ij: FloatArray | None,
        confidence: float,
        num_inliers: int,
        config: StitchingConfig,
) -> str | None:
    if affine_ij is None or not np.isfinite(affine_ij).all():
        return "No finite affine transform"
    if confidence < config.confidence_threshold:
        return "Confidence below threshold"
    if num_inliers < config.min_inliers:
        return "Too few RANSAC inliers"
    scales = np.linalg.svd(affine_ij[:2, :2], compute_uv=False)
    if scales[-1] <= config.min_scale or scales[0] >= config.max_scale:
        return "Affine scale outside allowed range"
    if scales[0] / scales[-1] >= config.max_anisotropy:
        return "Affine anisotropy outside allowed range"
    return None
