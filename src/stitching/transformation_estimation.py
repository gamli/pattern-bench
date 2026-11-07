from typing import Sequence, cast

import numpy as np
from scipy.optimize import least_squares

from cli.commands.utils import print_info, start_info_timer
from ._rigid_geometry import fit_rigid_transform
from .feature_matching import PairwiseMatch
from .pixel_registration import prepare_pixel_images, refine_global_pixels, validate_pixel_pairs
from .shared import StitchingConfig, StitchingImages, FloatArray


def estimate_global_transforms(
        pairwise_matches: Sequence[PairwiseMatch],
        stitching_images: StitchingImages,
        config: StitchingConfig,
) -> tuple[FloatArray, ...]:
    """Validate pairs, fit features jointly, then refine pixels with image 0 fixed.

    For each correspondence p_i <-> p_j, require G_i(p_i) = G_j(p_j).
    Each unknown rigid transform has one angle and two translations; scale
    is fixed to 1 and shear is excluded. Pairwise Kabsch fits along a spanning
    tree provide starting poses. All point correspondences are then optimized
    jointly with scipy.optimize.least_squares (MINPACK Levenberg-Marquardt),
    using an analytic Jacobian and Jacobian-based parameter scaling.
    Initial ordinary least squares is followed by the configured number of
    IRLS solves using radial Huber weights shared by both point coordinates.
    When enabled, independent pixel checks remove false feature pairs before
    that fit. A final joint intensity fit refines all remaining overlaps,
    keeping the same rigid model and ending at original image resolution.
    """

    print_info("Estimate Global Transformations")
    info_timer = start_info_timer()

    print_info("Checking Connectivity")

    _check_connectivity(pairwise_matches, stitching_images.count)

    info_timer.print(f"\tDone")

    if stitching_images.count == 1:
        return (np.eye(3, dtype=np.float64),)

    for pair in pairwise_matches:
        if not pair.accepted:
            continue
        p, q = pair.keypoints_i, pair.keypoints_j
        if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
            raise ValueError("Accepted pairs need point arrays with shape (N, 2), N >= 2.")
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("Matched coordinates must be finite.")

    print_info("Computing pixel pyramids")

    pixel_pyramids = None
    if config.pixel_refinement:
        pixel_pyramids = prepare_pixel_images(stitching_images)
        pairwise_matches = validate_pixel_pairs(pairwise_matches, pixel_pyramids, config)
        _check_connectivity(pairwise_matches, stitching_images.count)

    info_timer.print(f"\tDone")

    coefficient_blocks = []
    target_blocks = []
    for pair in pairwise_matches:
        if not pair.accepted:
            continue
        p, q = pair.keypoints_i, pair.keypoints_j
        coefficients = np.zeros((len(p), 3 * (stitching_images.count - 1)), dtype=np.float64)
        targets = np.zeros((len(p), 2), dtype=np.float64)
        i, j = pair.image_idx_i, pair.image_idx_j
        if i == 0:
            targets -= p
        else:
            coefficients[:, 3 * (i - 1):3 * i] = np.column_stack((p, np.ones(len(p))))
        if j == 0:
            targets += q
        else:
            coefficients[:, 3 * (j - 1):3 * j] = -np.column_stack((q, np.ones(len(q))))
        coefficient_blocks.append(coefficients)
        target_blocks.append(targets)

    coefficients = np.vstack(coefficient_blocks)
    targets = np.vstack(target_blocks)

    print_info("Computing Initial Kabsch Solution")

    solution = _initial_rigid_solution(pairwise_matches, stitching_images.count)

    info_timer.print(f"\tDone")

    print_info("Computing Initial Levenberg-Marquardt Solution")

    solution, rank = _solve_rigid(coefficients, targets, solution)

    info_timer.print(f"\tDone")

    print_info("Running IRLS iterations")

    for iteration in range(config.irls_iterations):
        errors = np.linalg.norm(coefficients @ solution - targets, axis=1)
        # sqrt(weight) multiplies the least-squares rows, not weight itself.
        weights = np.sqrt(np.minimum(1.0, config.huber_delta / np.maximum(errors, 1e-12)))
        solution, _ = _solve_rigid(coefficients, targets, solution, weights)

    info_timer.print(f"\tDone")

    transforms = [np.eye(3, dtype=np.float64)]
    for index in range(stitching_images.count - 1):
        transform = np.vstack((solution[3 * index:3 * index + 3].T, [0.0, 0.0, 1.0]))
        if not np.isfinite(transform).all() or abs(np.linalg.det(transform[:2, :2])) < 1e-12:
            message = f"Degenerate global rigid transform for image {index + 1}."
            raise RuntimeError(message)
        transforms.append(transform)

    if pixel_pyramids is not None:
        print_info("Running pixel based refinement")
        transforms = list(refine_global_pixels(transforms, pairwise_matches, pixel_pyramids, config))
        info_timer.print(f"\tDone")

    return tuple(transforms)


def _check_connectivity(matches: Sequence[PairwiseMatch], image_count: int) -> None:
    """Fail if any image cannot be linked to image 0 through accepted pairs."""

    if image_count < 1:
        raise ValueError("image_count must be positive.")

    adjacency: list[list[int]] = [[] for _ in range(image_count)]

    for pair in matches:
        if not 0 <= pair.image_idx_i < pair.image_idx_j < image_count:
            raise ValueError("Pair indices must satisfy 0 <= image_i < image_j < image_count.")
        if pair.accepted:
            adjacency[pair.image_idx_i].append(pair.image_idx_j)
            adjacency[pair.image_idx_j].append(pair.image_idx_i)

    reached = {0}
    pending = [0]

    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)

    missing = sorted(set(range(image_count)) - reached)

    if missing:
        raise RuntimeError(f"Disconnected match graph; unregistered image indices: {missing}.")


def _initial_rigid_solution(matches: Sequence[PairwiseMatch], image_count: int) -> FloatArray:
    """Initialize poses from pairwise Kabsch fits; the final fit uses all matches."""

    adjacency: list[list[tuple[int, FloatArray]]] = [[] for _ in range(image_count)]

    for pair in matches:
        if not pair.accepted:
            continue
        relative = fit_rigid_transform(pair.keypoints_j, pair.keypoints_i)
        rotation = relative[:2, :2]
        inverse = np.eye(3, dtype=np.float64)
        inverse[:2, :2] = rotation.T
        inverse[:2, 2] = -rotation.T @ relative[:2, 2]
        adjacency[pair.image_idx_i].append((pair.image_idx_j, relative))
        adjacency[pair.image_idx_j].append((pair.image_idx_i, inverse))

    poses: list[FloatArray | None] = [None] * image_count
    poses[0] = np.eye(3, dtype=np.float64)
    pending = [0]

    while pending:
        index = pending.pop()
        for neighbor, relative in adjacency[index]:
            if poses[neighbor] is None:
                poses[neighbor] = cast(FloatArray, poses[index]) @ relative
                pending.append(neighbor)

    return np.vstack([cast(FloatArray, pose)[:2].T for pose in poses[1:]])


def _solve_rigid(
        coefficients: FloatArray,
        targets: FloatArray,
        initial_solution: FloatArray,
        row_weights: FloatArray | None = None,
) -> tuple[FloatArray, int]:
    """Solve fixed-weight rigid least squares with SciPy's MINPACK LM solver."""

    initial_parameters = np.column_stack((
        np.arctan2(initial_solution[0::3, 1], initial_solution[0::3, 0]),
        initial_solution[2::3],
    )).ravel()

    weights = np.ones(len(targets)) if row_weights is None else row_weights

    x, y, translation = coefficients[:, 0::3], coefficients[:, 1::3], coefficients[:, 2::3]

    def make_solution(parameters: FloatArray) -> FloatArray:
        values = parameters.reshape(-1, 3)
        cosine, sine = np.cos(values[:, 0]), np.sin(values[:, 0])
        solution = np.empty_like(initial_solution)
        solution[0::3] = np.column_stack((cosine, sine))
        solution[1::3] = np.column_stack((-sine, cosine))
        solution[2::3] = values[:, 1:]
        return solution

    def residuals(parameters: FloatArray) -> FloatArray:
        return ((coefficients @ make_solution(parameters) - targets) * weights[:, None]).ravel()

    def jacobian(parameters: FloatArray) -> FloatArray:
        angles = parameters[0::3]
        cosine, sine = np.cos(angles), np.sin(angles)
        jac = np.zeros((len(targets), 2, len(parameters)), dtype=np.float64)
        jac[:, 0, 0::3] = -x * sine - y * cosine
        jac[:, 1, 0::3] = x * cosine - y * sine
        jac[:, 0, 1::3] = translation
        jac[:, 1, 2::3] = translation
        jac *= weights[:, None, None]
        return jac.reshape(-1, len(parameters))

    # Keep radial Huber weighting in outer IRLS; SciPy's Huber loss would
    # robustify x and y separately instead of using one weight per point.
    result = least_squares(
        residuals,
        initial_parameters,
        jac=jacobian,
        method="lm",
        x_scale="jac",
        loss="linear",
    )

    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
        raise RuntimeError(f"Rigid registration failed: {result.message}")

    rank = int(np.linalg.matrix_rank(result.jac))

    if rank != len(result.x):
        raise RuntimeError(f"Rigid registration is underdetermined: rank {rank}, expected {len(result.x)}.")

    return make_solution(result.x), rank
