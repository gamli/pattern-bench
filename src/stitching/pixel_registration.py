"""Validate feature pairs and jointly refine rigid poses using scan intensities.

Local contrast suppresses uniform backgrounds and slow illumination changes;
no background color or white threshold is assumed. Normalized contrast
differences require sufficient texture to validate a pair. Pairwise ECC at
coarse-to-fine resolutions checks each edge independently, before an incorrect
edge can bias the global fit. The final direct fit optimizes all remaining
overlaps together, ending at original resolution with image 0 fixed.
"""

from dataclasses import dataclass, replace
from typing import Sequence, cast

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_array, eye_array, hstack, vstack

from ._rigid_geometry import fit_rigid_transform
from .feature_matching import PairwiseMatch
from .shared import FloatArray, ImageArray, StitchingConfig, StitchingImages

PixelPyramid = tuple[ImageArray, ...]
_MIN_OVERLAP_PIXELS = 1024
_MIN_TEXTURE_PIXELS = 64


def prepare_pixel_images(images: StitchingImages) -> tuple[PixelPyramid, ...]:
    """Keep compact uint8 contrast pyramids; original scans are never changed."""
    pyramids: list[PixelPyramid] = []
    noise_floor = np.array([4.0], dtype=np.float64)
    for image in images.scan_images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        background = cv.GaussianBlur(gray, (0, 0), 12.0)
        ink = cv.subtract(cv.subtract(background, gray), noise_floor)
        pyramid: list[ImageArray] = [np.asarray(cv.GaussianBlur(ink, (5, 5), 0), dtype=np.uint8)]
        while len(pyramid) < 3 and min(pyramid[-1].shape) >= 128:
            pyramid.append(np.asarray(cv.pyrDown(pyramid[-1]), dtype=np.uint8))
        pyramids.append(tuple(pyramid))
    return tuple(pyramids)


def validate_pixel_pairs(
        matches: Sequence[PairwiseMatch],
        pyramids: Sequence[PixelPyramid],
        config: StitchingConfig,
) -> tuple[PairwiseMatch, ...]:
    """Reject edges lacking pixel evidence after a bounded local rigid fit."""

    verified, rows = [], []
    for pair in matches:
        if not pair.accepted:
            verified.append(pair)
            continue
        pyramid_i, pyramid_j = pyramids[pair.image_idx_i], pyramids[pair.image_idx_j]
        initial = fit_rigid_transform(pair.keypoints_i, pair.keypoints_j)
        initial_score = _pair_score(pyramid_i[0], pyramid_j[0], initial)
        refined = _refine_pair(initial, pyramid_i, pyramid_j, config)
        score = _pair_score(pyramid_i[0], pyramid_j[0], refined)
        if initial_score is not None and (score is None or initial_score > score):
            score = initial_score
        reason = None
        if score is None:
            reason = "Insufficient textured pixel overlap"
        elif score < config.pixel_min_correlation:
            reason = f"Pixel correlation {score:.4f} below {config.pixel_min_correlation:.4f}"
        if reason:
            pair = replace(
                pair, rejection_reason=reason,
                keypoints_i=np.empty((0, 2), dtype=np.float64),
                keypoints_j=np.empty((0, 2), dtype=np.float64),
            )
        verified.append(pair)
        rows.append((
            f"{pair.image_idx_i} / {pair.image_idx_j}",
            _format_score(initial_score), _format_score(score), reason or "Accepted",
        ))

    return tuple(verified)


def _overlap_view(
        reference: ImageArray,
        moving: ImageArray,
        transform: FloatArray,
        margin: int = 3,
) -> tuple[ImageArray, ImageArray, ImageArray, FloatArray] | None:
    """Crop to the intersection; transform maps reference pixels to moving pixels."""
    height, width = moving.shape
    inverse = np.linalg.inv(transform)
    corners = np.array(((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)))
    corners = corners @ inverse[:2, :2].T + inverse[:2, 2]
    lower = np.maximum(0, np.floor(corners.min(axis=0))).astype(int)
    upper = np.minimum(reference.shape[::-1], np.ceil(corners.max(axis=0)) + 1).astype(int)
    if np.any(upper - lower <= 2 * margin):
        return None
    x, y = lower
    right, bottom = upper
    template = np.ascontiguousarray(reference[y:bottom, x:right])
    warped, mask = _warp_view(moving, transform, template.shape, lower, margin)
    return template, warped, mask, lower.astype(np.float64)


def _warp_view(
        moving: ImageArray,
        transform: FloatArray,
        shape: tuple[int, int],
        origin: FloatArray,
        margin: int = 3,
) -> tuple[ImageArray, ImageArray]:
    sampling = transform[:2].copy()
    sampling[:, 2] += sampling[:, :2] @ origin
    warped = cast(
        ImageArray,
        cv.warpAffine(moving, sampling, shape[::-1], flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP),
    )
    mask = cast(
        ImageArray,
        cv.warpAffine(
            np.full(moving.shape, 255, dtype=np.uint8),
            sampling,
            shape[::-1],
            flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP,
        ),
    )
    mask = cast(
        ImageArray,
        cv.erode(
            mask,
            np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8),
            borderType=cv.BORDER_CONSTANT,
            borderValue=0,
        ),
    )
    return warped, mask


def _correlation(reference: ImageArray, moving: ImageArray, mask: ImageArray) -> float | None:
    if np.count_nonzero(((reference > 2) | (moving > 2)) & (mask != 0)) < _MIN_TEXTURE_PIXELS:
        return None
    norm_i = cv.norm(reference, cv.NORM_L2, mask)
    norm_j = cv.norm(moving, cv.NORM_L2, mask)
    if min(norm_i, norm_j) < 1e-6:
        return None
    # The norm identity computes the dot product without large float temporaries.
    # For normalized signals, correlation = 1 - 0.5 * squared difference.
    # Flat signals cannot establish a match even when their raw difference is zero.
    difference = cv.norm(reference, moving, cv.NORM_L2, mask)
    return float(np.clip((norm_i ** 2 + norm_j ** 2 - difference ** 2) / (2 * norm_i * norm_j), 0, 1))


def _pair_score(reference: ImageArray, moving: ImageArray, transform: FloatArray) -> float | None:
    view = _overlap_view(reference, moving, transform)
    if view is None:
        return None
    template, warped, mask, _ = view
    if cv.countNonZero(mask) < _MIN_OVERLAP_PIXELS:
        return None
    return _correlation(template, warped, mask)


def _refine_pair(
        initial: FloatArray,
        pyramid_i: PixelPyramid,
        pyramid_j: PixelPyramid,
        config: StitchingConfig,
) -> FloatArray:
    current = initial.copy()

    initial_view = _overlap_view(pyramid_i[0], pyramid_j[0], initial)
    if initial_view is None:
        return current

    shape, origin = initial_view[0].shape, initial_view[3]
    anchors = np.array(((0, 0), (shape[1] - 1, 0), (0, shape[0] - 1), (shape[1] - 1, shape[0] - 1))) + origin

    for level in reversed(range(min(len(pyramid_i), len(pyramid_j)))):

        scale = 0.5 ** level
        scaled = current.copy()
        scaled[:2, 2] *= scale

        view = _overlap_view(
            pyramid_i[level],
            pyramid_j[level],
            scaled,
            margin=3 + int(np.ceil(config.pixel_max_correction * scale)),
        )

        if view is None or cv.countNonZero(view[2]) < 128:
            continue

        template, moving, mask, origin = view
        before = _correlation(template, moving, mask)

        if before is None:
            continue

        try:
            _, correction = cv.findTransformECC(
                template,
                moving,
                np.eye(2, 3, dtype=np.float32),
                cv.MOTION_EUCLIDEAN,
                (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, config.pixel_refinement_iterations, 1e-5),
                mask,
                5,
            )
        except cv.error:  # todo does this try/except make sense?
            continue

        if not np.isfinite(correction).all():
            continue

        # pre-warping keeps ECC near identity, including for scans rotated 90 degrees.

        angle = np.arctan2(correction[1, 0], correction[0, 0])
        cosine, sine = np.cos(angle), np.sin(angle)

        delta = np.eye(3, dtype=np.float64)
        delta[:2, :2] = ((cosine, -sine), (sine, cosine))
        delta[:2, 2] = correction[:, 2]

        offset = np.eye(3, dtype=np.float64)
        offset[:2, 2] = origin

        candidate = scaled @ offset @ delta @ np.linalg.inv(offset)
        candidate[:2, 2] /= scale

        displacement = anchors @ (candidate[:2, :2] - initial[:2, :2]).T + candidate[:2, 2] - initial[:2, 2]

        if np.max(np.linalg.norm(displacement, axis=1)) > config.pixel_max_correction:
            continue

        candidate_scaled = candidate.copy()
        candidate_scaled[:2, 2] *= scale

        warped, candidate_mask = _warp_view(pyramid_j[level], candidate_scaled, template.shape, origin)

        common_mask = cast(ImageArray, cv.bitwise_and(mask, candidate_mask))

        old_score = _correlation(template, moving, common_mask)
        new_score = _correlation(template, warped, common_mask)

        if old_score is not None and new_score is not None and new_score > old_score:
            current = candidate

    return current


@dataclass(frozen=True)
class _PixelSamples:
    image_i: int
    image_j: int
    points_i: FloatArray
    reference: FloatArray
    moving: ImageArray
    scale: float


def _make_samples(
        matches: Sequence[PairwiseMatch],
        pyramids: Sequence[PixelPyramid],
        poses: Sequence[FloatArray],
        level: int,
        config: StitchingConfig,
) -> tuple[_PixelSamples, ...]:
    samples = []
    scale = 0.5 ** level

    for pair in matches:

        if not pair.accepted:
            continue

        i, j = pair.image_idx_i, pair.image_idx_j

        if level >= min(len(pyramids[i]), len(pyramids[j])):
            continue

        pose_j = poses[j]
        pose_j_inv = np.linalg.inv(pose_j)
        pose_i = poses[i]
        relative = pose_j_inv @ pose_i
        relative[:2, 2] *= scale

        margin = 3 + int(np.ceil(2 * config.pixel_max_correction * scale))

        view = _overlap_view(pyramids[i][level], pyramids[j][level], relative, margin)

        if view is None:
            continue

        reference, moving, mask, origin = view

        # Include contrast from either scan and nearby areas, so unmatched lines count too.

        radius = max(3, int(np.ceil(config.pixel_max_correction * scale)))

        ink = ((reference > 2) | (moving > 2)).astype(np.uint8)
        ink = cv.dilate(ink, np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8))

        indices = np.flatnonzero((ink != 0) & (mask != 0))

        if len(indices) < 128:
            continue

        if len(indices) > config.pixel_max_samples:
            rng = np.random.default_rng(0x5A17 + 1009 * i + 9176 * j + level)
            indices = rng.choice(indices, config.pixel_max_samples, replace=False)

        y, x = np.unravel_index(indices, reference.shape)

        values = reference[y, x].astype(np.float64)

        norm = np.linalg.norm(values)

        if norm < 1e-6:
            continue

        points = (np.column_stack((x, y)) + origin) / scale

        samples.append(_PixelSamples(i, j, points, values / norm, pyramids[j][level], scale))

    return tuple(samples)


def _bilinear(image: ImageArray, coordinates: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Sample intensities and their exact bilinear derivatives without remap quantization."""
    height, width = image.shape
    valid = ((coordinates >= 0) & (coordinates < (width - 1, height - 1))).all(axis=1)
    clipped = np.clip(coordinates, (0, 0), (width - 1.000001, height - 1.000001))
    x, y = clipped.T
    left, top = np.floor(clipped).astype(int).T
    dx, dy = x - left, y - top
    a = image[top, left].astype(np.float64)
    b = image[top, left + 1].astype(np.float64)
    c = image[top + 1, left].astype(np.float64)
    d = image[top + 1, left + 1].astype(np.float64)
    values = (a * (1 - dx) + b * dx) * (1 - dy) + (c * (1 - dx) + d * dx) * dy
    gradients = np.column_stack(((b - a) * (1 - dy) + (d - c) * dy, (c - a) * (1 - dx) + (d - b) * dx))
    values[~valid] = 0
    gradients[~valid] = 0
    return values, gradients


class _PixelObjective:
    """Direct normalized-intensity residuals with an analytic pose Jacobian."""

    def __init__(
            self, base: Sequence[FloatArray], samples: Sequence[_PixelSamples],
            centres: FloatArray, radii: FloatArray, max_correction: float,
    ) -> None:
        self.base = base
        self.samples = samples
        self.centres = centres
        self.radii = radii
        self.prior_weight = 1e-3 / max_correction
        self._parameters: FloatArray | None = None
        self._residuals = np.empty(0, dtype=np.float64)
        self._jacobian = csr_array((0, 0), dtype=np.float64)

    def poses(self, parameters: FloatArray) -> tuple[FloatArray, ...]:
        poses = [self.base[0].copy()]
        for index, values in enumerate(parameters.reshape(-1, 3), start=1):
            angle = values[0] / self.radii[index]
            cosine, sine = np.cos(angle), np.sin(angle)
            correction = np.eye(3, dtype=np.float64)
            correction[:2, :2] = ((cosine, -sine), (sine, cosine))
            correction[:2, 2] = self.centres[index] + values[1:] - correction[:2, :2] @ self.centres[index]
            poses.append(self.base[index] @ correction)
        return tuple(poses)

    def _derivatives(self, index: int, points: FloatArray, parameters: FloatArray) -> FloatArray:
        angle = parameters[3 * (index - 1)] / self.radii[index]
        cosine, sine = np.cos(angle), np.sin(angle)
        derivative = self.base[index][:2, :2] @ np.array(((-sine, -cosine), (cosine, -sine)))
        result = np.empty((len(points), 2, 3), dtype=np.float64)
        result[:, :, 0] = (points - self.centres[index]) @ derivative.T / self.radii[index]
        result[:, :, 1:] = self.base[index][:2, :2]
        return result

    def _evaluate(self, parameters: FloatArray) -> None:
        if self._parameters is not None and np.array_equal(parameters, self._parameters):
            return
        poses = self.poses(parameters)
        row_count = sum(len(sample.reference) for sample in self.samples)
        residuals = np.empty(row_count + len(parameters), dtype=np.float64)
        jacobian_blocks = []
        offset = 0
        for sample in self.samples:
            i, j = sample.image_i, sample.image_j
            relative = np.linalg.inv(poses[j]) @ poses[i]
            points_j = sample.points_i @ relative[:2, :2].T + relative[:2, 2]
            values, gradients = _bilinear(sample.moving, points_j * sample.scale)
            gradients *= sample.scale
            norm = max(float(np.linalg.norm(values)), 1e-12)
            rows = slice(offset, offset + len(values))
            residuals[rows] = values / norm - sample.reference
            blocks = []
            next_column = 0
            for index, points, sign in ((i, sample.points_i, 1), (j, points_j, -1)):
                if index == 0:
                    continue
                derivatives = self._derivatives(index, points, parameters)
                derivatives = np.einsum("ab,nbc->nac", poses[j][:2, :2].T, derivatives) * sign
                derivatives = np.einsum("na,nab->nb", gradients, derivatives)
                normalized = derivatives / norm - np.outer(values, values @ derivatives) / norm ** 3
                first_column = 3 * (index - 1)
                blocks.extend((csr_array((len(values), first_column - next_column)), csr_array(normalized)))
                next_column = first_column + 3
            blocks.append(csr_array((len(values), len(parameters) - next_column)))
            jacobian_blocks.append(hstack(blocks, format="csr"))
            offset += len(values)
        # A weak prior retains the feature pose in directions without pixel texture.
        residuals[row_count:] = parameters * self.prior_weight
        jacobian_blocks.append(eye_array(len(parameters), format="csr") * self.prior_weight)
        self._parameters = parameters.copy()
        self._residuals, self._jacobian = residuals, vstack(jacobian_blocks, format="csr")

    def residuals(self, parameters: FloatArray) -> FloatArray:
        self._evaluate(parameters)
        return self._residuals

    def jacobian(self, parameters: FloatArray) -> csr_array:
        self._evaluate(parameters)
        return self._jacobian


def refine_global_pixels(
        transforms: Sequence[FloatArray],
        matches: Sequence[PairwiseMatch],
        pyramids: Sequence[PixelPyramid],
        config: StitchingConfig,
) -> tuple[FloatArray, ...]:
    """Jointly maximize pixel agreement, keeping unit scale, rigid poses and image 0."""

    centres = np.array(
        [(pyramid[0].shape[1] - 1, pyramid[0].shape[0] - 1) for pyramid in pyramids],
        dtype=np.float64,
    ) / 2
    radii = np.maximum(np.linalg.norm(centres, axis=1), 1.0)
    parameters = np.zeros(3 * (len(transforms) - 1), dtype=np.float64)
    poses = tuple(transforms)
    refined_full_resolution = False

    for level in reversed(range(max(len(pyramid) for pyramid in pyramids))):
        samples = _make_samples(matches, pyramids, poses, level, config)
        if not samples:
            continue
        objective = _PixelObjective(transforms, samples, centres, radii, config.pixel_max_correction)
        result = least_squares(
            objective.residuals, parameters, jac=objective.jacobian, method="trf",
            bounds=(-config.pixel_max_correction, config.pixel_max_correction),
            x_scale="jac", max_nfev=config.pixel_refinement_iterations,
            ftol=1e-6, xtol=1e-5, gtol=1e-7,
        )
        if np.isfinite(result.x).all() and np.isfinite(result.fun).all():
            parameters = result.x
            poses = objective.poses(parameters)
            refined_full_resolution |= level == 0

    movement = _maximum_pose_movement(transforms, poses, pyramids)

    if movement > config.pixel_max_correction:
        # Limit actual corner motion, including the combined rotation/translation.
        lower, upper = 0.0, 1.0
        for _ in range(24):
            fraction = (lower + upper) / 2
            # noinspection unbound-local-variable
            candidate = objective.poses(parameters * fraction)
            if _maximum_pose_movement(transforms, candidate, pyramids) <= config.pixel_max_correction:
                lower = fraction
            else:
                upper = fraction
        poses = objective.poses(parameters * lower)

    before_scores: list[float] = []
    after_scores: list[float] = []

    for pair in matches:

        if not pair.accepted:
            continue

        i, j = pair.image_idx_i, pair.image_idx_j

        relative_before = np.asarray(np.linalg.inv(transforms[j]) @ transforms[i], dtype=np.float64)
        relative_after = np.asarray(np.linalg.inv(poses[j]) @ poses[i], dtype=np.float64)

        before = _pair_score(pyramids[i][0], pyramids[j][0], relative_before)
        after = _pair_score(pyramids[i][0], pyramids[j][0], relative_after)

        before_scores.append(before if before is not None else 0.0)
        after_scores.append(after if after is not None else 0.0)

    improved = (refined_full_resolution
                and np.mean(after_scores) > np.mean(before_scores) + 1e-6
                and min(after_scores) >= config.pixel_min_correlation)

    return poses if improved else tuple(transforms)


def _maximum_pose_movement(
        before: Sequence[FloatArray], after: Sequence[FloatArray], pyramids: Sequence[PixelPyramid],
) -> float:
    maximum = 0.0
    for initial, refined, pyramid in zip(before, after, pyramids):
        height, width = pyramid[0].shape
        corners = np.array(((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)))
        movement = corners @ (refined[:2, :2] - initial[:2, :2]).T + refined[:2, 2] - initial[:2, 2]
        maximum = max(maximum, float(np.max(np.linalg.norm(movement, axis=1))))
    return maximum


def _format_score(score: float | None) -> str:
    return "Insufficient texture/overlap" if score is None else f"{score:.4f}"
