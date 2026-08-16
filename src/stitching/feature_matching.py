import itertools
from dataclasses import dataclass
from typing import Sequence

import cv2 as cv
import numpy as np

from src.stitching.shared import FloatArray, StitchingConfig, StitchingImages
from utils.md_logger import MdLogger


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

    vlog = config.vlog

    image_count = stitching_images.count

    pair_count = image_count * (image_count - 1) // 2

    if vlog:
        vlog.h3("Feature Matching")
        vlog.start_timer(f"Feature Matching (total)")
        vlog.log_elapsed("Feature Matching starting at")
        vlog.p(f"Matching features on `{image_count}` images across `{pair_count}` image pairs.")

    if image_count == 1:
        if vlog:
            vlog.p("Only one image was provided; there are no image pairs to match.")
            vlog.end_timer()
            vlog.log_elapsed("Feature Matching finished at")
        return ()

    if vlog:
        vlog.p("**Matching Config**")
        vlog.table(
            ("Parameter", "Value"),
            (
                ("match_conf", config.match_conf),
                ("num_matches_thresh1", config.num_matches_thresh1),
                ("confidence_threshold", config.confidence_threshold),
                ("min_inliers", config.min_inliers),
                ("min_scale", config.min_scale),
                ("max_scale", config.max_scale),
                ("max_anisotropy", config.max_anisotropy),
            ),
        )
        vlog.p("Running bidirectional descriptor matching and full affine RANSAC.")

    if vlog:
        vlog.start_timer("matcher = cv.detail.AffineBestOf2NearestMatcher(...)")
        vlog.p("**AffineBestOf2NearestMatcher config**")
        vlog.table(
            ("Parameter", "Value"),
            (
                ("full_affine", True),
                ("try_use_gpu", config.use_gpu),
                ("match_conf", config.match_conf),
                ("num_matches_thresh1", config.num_matches_thresh1),
            ),
        )
    matcher = cv.detail.AffineBestOf2NearestMatcher(
        full_affine=True,
        try_use_gpu=config.use_gpu,
        match_conf=config.match_conf,
        num_matches_thresh1=config.num_matches_thresh1,
    )
    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.start_timer(f"config.apply_fixed_seed(...)")
    config.apply_fixed_seed()
    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.start_timer(f"Descriptor matching and affine RANSAC for all image pairs")

    if vlog:
        vlog.start_timer(f"pair_match_infos = matcher.apply2(...)")
    all_pair_matches_infos = matcher.apply2(image_features)
    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.start_timer(f"matcher.collectGarbage(...)")
    # just to release buffers and stuff early to save a little on memory
    matcher.collectGarbage()
    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.end_timer()

    keypoints = [feature.getKeypoints() for feature in image_features]

    pairs = []

    if vlog:
        vlog.start_timer(f"Evaluating/Visualizing all pair matches")
        vlog.p("Evaluating image pair match infos and visualizing their matches.")

    for i, j in itertools.combinations(range(image_count), 2):

        if vlog:
            vlog.start_timer(f"Evaluating/Visualizing pair {i} <-> {j} matches")

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

        if vlog:
            vlog.start_timer(f"Logging image pair {i} <-> {j} with matches")
        if vlog:
            _log_pair_matches(
                vlog,
                image_features[i],
                image_features[j],
                stitching_images,
                matches_info,
                pair,
            )
        if vlog:
            vlog.end_timer()

        if vlog:
            vlog.end_timer()

    accepted = [pair for pair in pairs if pair.accepted]

    if vlog:
        vlog.end_timer()
        vlog.p(
            "**Matching result:** "
            f"Accepted `{len(accepted)}` of `{pair_count}` image pairs with "
            f"`{sum(len(pair.keypoints_i) for pair in accepted)}` inlier correspondences; "
            f"rejected `{pair_count - len(accepted)}` pairs."
        )
        vlog.end_timer()
        vlog.log_elapsed("Feature Matching finished at")

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


# Bound affine-overlay allocations: about 300 MB per uint8 BGR canvas.
_MAX_AFFINE_OVERLAY_PIXELS = 100_000_000
_MAX_AFFINE_OVERLAY_SIDE = 30_000


def _log_pair_matches(
        vlog: MdLogger,
        features_i: cv.detail.ImageFeatures,
        features_j: cv.detail.ImageFeatures,
        stitching_images: StitchingImages,
        matches_info: cv.detail.MatchesInfo,
        pairwise_match: PairwiseMatch,
) -> None:
    """Preview accepted inliers; link other match views."""
    file_i = stitching_images.scan_files[features_i.img_idx]
    file_j = stitching_images.scan_files[features_j.img_idx]

    vlog.h4(f"Pair {pairwise_match.image_idx_i} <-> {pairwise_match.image_idx_j}")
    vlog.p(f"Source: `{file_i}`; target: `{file_j}`.")
    if pairwise_match.accepted:
        vlog.table(
            ("Descriptor matches", "RANSAC inliers", "Confidence"),
            ((pairwise_match.num_matches, pairwise_match.num_inliers, f"{pairwise_match.confidence:.6f}"),),
        )
        if pairwise_match.affine_ij is not None:
            vlog.p("Pairwise affine transform from source to target:")
            vlog.code(np.array2string(pairwise_match.affine_ij, precision=8))
        vlog.p("**Accepted.**")
    else:
        vlog.err(f"Pair rejected: {pairwise_match.rejection_reason}.")

    matches = matches_info.getMatches()

    inliers = np.asarray(matches_info.getInliers(), dtype=bool).reshape(-1)

    inlier_matches: Sequence[cv.DMatch] = ()
    match_views: list[tuple[str, Sequence[cv.DMatch], bool]] = []
    if matches and len(inliers) == len(matches):
        inlier_matches = tuple(match for match, is_inlier in zip(matches, inliers) if is_inlier)
        match_views.append(("RANSAC inliers", inlier_matches, not pairwise_match.accepted))
    elif pairwise_match.accepted:
        vlog.p("No complete RANSAC inlier mask is available; descriptor matches are linked below.")

    # An empty match set still produces a useful side-by-side image link.
    match_views.append(("Descriptor matches", matches, True))

    image_i = stitching_images.scan_images[features_i.img_idx]
    image_j = stitching_images.scan_images[features_j.img_idx]

    keypoints_i = features_i.getKeypoints()
    keypoints_j = features_j.getKeypoints()

    height_i: int = image_i.shape[0]
    height_j: int = image_j.shape[0]

    width_i: int = image_i.shape[1]

    output_height: int = max(height_i, height_j)
    output_width: int = width_i + image_j.shape[1]

    output_image = np.empty((output_height, output_width, 3), dtype=np.uint8)

    affine_ij = pairwise_match.affine_ij
    affine_overlay = None
    overlay_caption = "affine transformed"
    target_height: int = image_j.shape[0]
    target_width: int = image_j.shape[1]
    overlay_height: int = target_height
    overlay_width: int = target_width
    points_i = np.array([point.pt for point in keypoints_i], dtype=np.float64).reshape(-1, 2)
    points_j = np.array([point.pt for point in keypoints_j], dtype=np.float64).reshape(-1, 2)
    if affine_ij is not None and np.isfinite(affine_ij).all():
        corners_i = np.array(((0, 0), (width_i, 0), (width_i, height_i), (0, height_i)), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            corners_i = corners_i @ affine_ij[:2, :2].T + affine_ij[:2, 2]
        corners = np.vstack((corners_i, ((0, 0), (target_width, target_height))))
        if np.isfinite(corners).all():
            minimum = np.floor(corners.min(axis=0))
            maximum = np.ceil(corners.max(axis=0))
            canvas_width: int = int(maximum[0] - minimum[0])
            canvas_height: int = int(maximum[1] - minimum[1])
            display_scale: float = min(
                1.0,
                float(np.sqrt(_MAX_AFFINE_OVERLAY_PIXELS / (canvas_width * canvas_height))),
                _MAX_AFFINE_OVERLAY_SIDE / max(canvas_width, canvas_height),
            )
            overlay_width = max(1, min(_MAX_AFFINE_OVERLAY_SIDE, int(np.ceil(canvas_width * display_scale))))
            overlay_height = max(1, min(_MAX_AFFINE_OVERLAY_SIDE, int(np.ceil(canvas_height * display_scale))))
            offset_x: int = -int(minimum[0])
            offset_y: int = -int(minimum[1])

            # Apply one display scale and offset to both images and their match coordinates.
            display_affine = affine_ij[:2].copy()
            display_affine[:, 2] -= minimum
            display_affine *= display_scale
            affine_overlay = cv.warpAffine(image_i, display_affine, (overlay_width, overlay_height))
            warped_mask = cv.warpAffine(
                np.full(image_i.shape[:2], 255, dtype=np.uint8),
                display_affine,
                (overlay_width, overlay_height),
                flags=cv.INTER_NEAREST,
            )
            if display_scale == 1.0:
                target_region = affine_overlay[offset_y:offset_y + target_height, offset_x:offset_x + target_width]
                overlap = warped_mask[offset_y:offset_y + target_height, offset_x:offset_x + target_width] != 0
                blended_image = cv.addWeighted(target_region, 0.5, image_j, 0.5, 0.0)
                target_region[overlap] = blended_image[overlap]
                target_region[~overlap] = image_j[~overlap]
            else:
                target_affine = np.array(
                    ((display_scale, 0.0, -minimum[0] * display_scale),
                     (0.0, display_scale, -minimum[1] * display_scale)),
                    dtype=np.float64,
                )
                target_image = cv.warpAffine(image_j, target_affine, (overlay_width, overlay_height))
                target_mask = cv.warpAffine(
                    np.full(image_j.shape[:2], 255, dtype=np.uint8),
                    target_affine,
                    (overlay_width, overlay_height),
                    flags=cv.INTER_NEAREST,
                )
                overlap = (warped_mask != 0) & (target_mask != 0)
                blended_image = cv.addWeighted(affine_overlay, 0.5, target_image, 0.5, 0.0)
                affine_overlay[overlap] = blended_image[overlap]
                target_only = (target_mask != 0) & (warped_mask == 0)
                affine_overlay[target_only] = target_image[target_only]
                overlay_caption += f" [canvas {canvas_width} x {canvas_height}; display scale {display_scale:.6g}]"

            with np.errstate(over="ignore", invalid="ignore"):
                points_i = (points_i @ affine_ij[:2, :2].T + affine_ij[:2, 2] - minimum) * display_scale
            points_j = (points_j - minimum) * display_scale

    for label, matches, link_only in match_views:

        if link_only:
            vlog.br()

        vlog.img(
            cv.drawMatches(
                image_i,
                keypoints_i,
                image_j,
                keypoints_j,
                matches,
                output_image,
                matchColor=(0, 255, 0),
                singlePointColor=(0, 255, 0),
                flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
            ),
            link_only=link_only,
            alt=label,
            style="max-height: 500px;",
        )

        if affine_overlay is not None:
            overlay_image = affine_overlay.copy()
            for match in matches:
                point_i = points_i[match.queryIdx]
                point_j = points_j[match.trainIdx]
                # OpenCV's line endpoints must fit signed 32-bit coordinates.
                if not np.isfinite(point_i).all() or np.max(np.abs(point_i)) > 2 ** 30:
                    continue
                cv.line(
                    overlay_image,
                    (round(float(point_i[0])), round(float(point_i[1]))),
                    (round(float(point_j[0])), round(float(point_j[1]))),
                    (0, 255, 0),
                    thickness=1,
                    lineType=cv.LINE_AA,
                )

            if link_only:
                vlog.br()

            vlog.img(
                overlay_image,
                link_only=link_only,
                alt=f"{label}, {overlay_caption}",
                style="max-height: 500px;",
            )
