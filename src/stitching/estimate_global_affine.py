from typing import Sequence, cast

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

from src.stitching.feature_matching import PairwiseMatch
from src.stitching.shared import StitchingConfig, StitchingImages, FloatArray
from utils.md_logger import MdLogger


def estimate_global_affine(
        pairwise_matches: Sequence[PairwiseMatch],
        stitching_images: StitchingImages,
        config: StitchingConfig,
) -> tuple[FloatArray, ...]:
    """Solve all accepted point correspondences jointly, fixing image 0 to identity.

    For each correspondence p_i <-> p_j, require G_i(p_i) = G_j(p_j).
    Each unknown rigid transform has one angle and two translations; scale
    is fixed to 1 and shear is excluded. Pairwise Kabsch fits along a spanning
    tree provide starting poses. All point correspondences are then optimized
    jointly with scipy.optimize.least_squares (MINPACK Levenberg-Marquardt),
    using an analytic Jacobian and Jacobian-based parameter scaling.
    Initial ordinary least squares is followed by the configured number of
    IRLS solves using radial Huber weights shared by both point coordinates.
    """

    vlog = config.vlog

    if vlog:
        vlog.h3("Global Rigid Estimation")
        vlog.start_timer(f"Global Rigid Estimation (total)")
        vlog.log_elapsed("Global Rigid Estimation starting at")
        vlog.p(f"Registering `{stitching_images.count}` images jointly; image `0` is fixed to identity.")
        vlog.table(
            ("Parameter", "Value"),
            (
                ("irls_iterations", config.irls_iterations),
                ("huber_delta (px)", config.huber_delta),
                ("solver", "SciPy least_squares / Levenberg-Marquardt"),
                ("initialization", "Pairwise Kabsch / spanning tree"),
            ),
        )

    try:
        if vlog:
            vlog.start_timer(f"check_connectivity(...)")
        check_connectivity(pairwise_matches, stitching_images.count)
        if vlog:
            vlog.end_timer()
    except (ValueError, RuntimeError) as error:
        if vlog:
            vlog.err(str(error))
        raise

    if stitching_images.count == 1:
        if vlog:
            vlog.p("Only one image was provided; returning identity without least squares or IRLS.")
            vlog.code(np.array2string(np.eye(3, dtype=np.float64), precision=8))
            vlog.end_timer()
            vlog.log_elapsed("Global Rigid Estimation finished at")
        return (np.eye(3, dtype=np.float64),)

    if vlog:
        vlog.start_timer(f"coefficient/target initialization")

    coefficient_blocks = []
    target_blocks = []
    for pair in pairwise_matches:
        if not pair.accepted:
            continue
        p, q = pair.keypoints_i, pair.keypoints_j
        if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
            if vlog:
                vlog.err(f"Pair {pair.image_idx_i} / {pair.image_idx_j}: invalid matched point array shapes.")
                vlog.end_timer()
                vlog.log_elapsed("Global Rigid Estimation finished at")
            raise ValueError("Accepted pairs need point arrays with shape (N, 2), N >= 2.")
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            if vlog:
                vlog.err(f"Pair {pair.image_idx_i} / {pair.image_idx_j}: matched coordinates must be finite.")
                vlog.end_timer()
                vlog.log_elapsed("Global Rigid Estimation finished at")
            raise ValueError("Matched coordinates must be finite.")
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

    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.p(
            f"The accepted match graph is connected. Using `{len(coefficient_blocks)}` accepted pairs "
            f"and `{len(targets)}` point correspondences; `{len(pairwise_matches) - len(coefficient_blocks)}` pairs are rejected."
        )
        vlog.p(
            f"The coordinate coefficient matrix has shape `{coefficients.shape}`. "
            f"The rigid fit has `{coefficients.shape[1]}` free parameters "
            f"and `{2 * len(targets)}` scalar residuals (angle, tx, ty per image except image 0)."
        )
        vlog.p(
            "Every step reports residual distances in the reference image's pixel coordinates. "
            "The unweighted sum of squared distances and the Huber cost use the same definitions in every step: "
            "Huber cost is `0.5 * r**2` for `r <= huber_delta`, otherwise `huber_delta * (r - 0.5 * huber_delta)`. "
            "IRLS point weights come from the preceding solution; residuals below are measured after each new solve."
        )

    if vlog:
        vlog.start_timer(f"Kabsch initialization")
    solution = _initial_rigid_solution(pairwise_matches, stitching_images.count)
    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.start_timer(f"Initial Levenberg-Marquardt")
    solution, rank = _solve_rigid(coefficients, targets, solution, vlog=vlog)
    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.p(f"The rigid-fit Jacobian has full column rank: `{rank}`.")
        _log_estimation_step(
            vlog, "Initial least squares", solution,
            np.linalg.norm(coefficients @ solution - targets, axis=1),
            pairwise_matches, stitching_images, config.huber_delta,
        )

    for iteration in range(config.irls_iterations):

        if vlog:
            vlog.start_timer(f"IRLS iteration {iteration + 1} / {config.irls_iterations}")

        if vlog:
            vlog.start_timer(f"np.linalg.norm(...)")
        errors = np.linalg.norm(coefficients @ solution - targets, axis=1)
        if vlog:
            vlog.end_timer()

        if vlog:
            vlog.start_timer(f"np.sqrt(...)")
        # sqrt(weight) multiplies the least-squares rows, not weight itself.
        weights = np.sqrt(np.minimum(1.0, config.huber_delta / np.maximum(errors, 1e-12)))
        if vlog:
            vlog.end_timer()

        if vlog:
            vlog.start_timer(f"Weighted Levenberg-Marquardt")
        solution, _ = _solve_rigid(coefficients, targets, solution, weights, vlog)
        if vlog:
            vlog.end_timer()

        if vlog:
            _log_estimation_step(
                vlog, f"IRLS iteration {iteration + 1} / {config.irls_iterations}", cast(FloatArray, solution),
                np.linalg.norm(coefficients @ solution - targets, axis=1),
                pairwise_matches, stitching_images, config.huber_delta, weights,
            )
            vlog.end_timer()

    transforms = [np.eye(3, dtype=np.float64)]
    for index in range(stitching_images.count - 1):
        transform = np.vstack((solution[3 * index:3 * index + 3].T, [0.0, 0.0, 1.0]))
        if not np.isfinite(transform).all() or abs(np.linalg.det(transform[:2, :2])) < 1e-12:
            if vlog:
                vlog.err(f"Degenerate global rigid transform for image {index + 1}.")
            raise RuntimeError(f"Degenerate global rigid transform for image {index + 1}.")
        transforms.append(transform)

    if vlog:
        vlog.h4("Global rigid estimation result")
        vlog.p(f"Returning `{len(transforms)}` global transforms mapping each image to reference image `0`.")
        vlog.end_timer()
        vlog.log_elapsed("Global Rigid Estimation finished at")

    return tuple(transforms)


def _initial_rigid_solution(matches: Sequence[PairwiseMatch], image_count: int) -> FloatArray:
    """Initialize poses from pairwise Kabsch fits; the final fit uses all matches."""
    adjacency: list[list[tuple[int, FloatArray]]] = [[] for _ in range(image_count)]
    for pair in matches:
        if not pair.accepted:
            continue
        p, q = pair.keypoints_i, pair.keypoints_j
        p_mean, q_mean = p.mean(axis=0), q.mean(axis=0)
        u, _, vt = np.linalg.svd((q - q_mean).T @ (p - p_mean))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        relative = np.eye(3, dtype=np.float64)
        relative[:2, :2] = rotation
        relative[:2, 2] = p_mean - rotation @ q_mean
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
        vlog: MdLogger | None = None,
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
        result = np.zeros((len(targets), 2, len(parameters)), dtype=np.float64)
        result[:, 0, 0::3] = -x * sine - y * cosine
        result[:, 1, 0::3] = x * cosine - y * sine
        result[:, 0, 1::3] = translation
        result[:, 1, 2::3] = translation
        result *= weights[:, None, None]
        return result.reshape(-1, len(parameters))

    # Keep radial Huber weighting in outer IRLS; SciPy's Huber loss would
    # robustify x and y separately instead of using one weight per point.
    result = least_squares(
        residuals, initial_parameters, jac=jacobian,
        method="lm", x_scale="jac", loss="linear",
    )
    if vlog:
        vlog.table(
            ("Solver metric", "Value"),
            (
                ("Termination", result.message),
                ("Function evaluations", result.nfev),
                ("Jacobian evaluations", result.njev),
                ("First-order optimality", f"{result.optimality:.6g}"),
            ),
        )
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
        message = f"Rigid registration failed: {result.message}"
        if vlog:
            vlog.err(message)
        raise RuntimeError(message)
    rank = int(np.linalg.matrix_rank(result.jac))
    if rank != len(result.x):
        message = f"Rigid registration is underdetermined: rank {rank}, expected {len(result.x)}."
        if vlog:
            vlog.err(message)
        raise RuntimeError(message)
    return make_solution(result.x), rank


def _log_estimation_step(
        vlog: MdLogger,
        title: str,
        solution: FloatArray,
        errors: FloatArray,
        matches: Sequence[PairwiseMatch],
        stitching_images: StitchingImages,
        huber_delta: float,
        row_weights: FloatArray | None = None,
) -> None:
    """Report the new solution and the optional row weights used for this solve."""

    vlog.start_timer(f"_log_estimation_step(...)")

    squared_errors = errors ** 2
    clipped_errors = np.minimum(errors, huber_delta)
    huber_cost = np.sum(0.5 * clipped_errors ** 2 + huber_delta * (errors - clipped_errors))
    point_weights = row_weights ** 2 if row_weights is not None else np.ones_like(errors)

    vlog.h4(title)
    vlog.table(
        ("Metric", "Value"),
        (
            ("Unweighted sum of squared distances (px^2)", f"{np.sum(squared_errors):.8g}"),
            ("Huber cost (px^2)", f"{huber_cost:.8g}"),
            ("Point RMSE (px)", f"{np.sqrt(np.mean(squared_errors)):.4f}"),
            ("Median distance (px)", f"{np.median(errors):.4f}"),
            ("95th percentile distance (px)", f"{np.percentile(errors, 95):.4f}"),
            ("Maximum distance (px)", f"{np.max(errors):.4f}"),
            ("Residuals above Huber delta after solve", f"{np.count_nonzero(errors > huber_delta)} / {len(errors)}"),
            ("Downweighted points used in this solve", f"{np.count_nonzero(point_weights < 1.0)} / {len(errors)}"),
            ("Minimum point weight used in this solve", f"{np.min(point_weights):.6f}"),
        ),
    )

    pair_rows = []
    offset = 0
    for pair in matches:
        if not pair.accepted:
            continue
        count = len(pair.keypoints_i)
        pair_errors = errors[offset:offset + count]
        pair_rows.append((
            f"{pair.image_idx_i} / {pair.image_idx_j}", count,
            f"{np.sqrt(np.mean(pair_errors ** 2)):.4f}",
            f"{np.percentile(pair_errors, 95):.4f}",
            f"{np.max(pair_errors):.4f}",
        ))
        offset += count

    vlog.h5("Residuals per accepted pair")
    vlog.table(("Image pair", "Points", "RMSE (px)", "P95 (px)", "Maximum (px)"), pair_rows)
    vlog.h5("Global transforms (image -> reference image 0)")
    vlog.p("Image `0` stays at identity.")
    transforms = [np.eye(3, dtype=np.float64)]
    for index in range(solution.shape[0] // 3):
        transform = np.vstack((solution[3 * index:3 * index + 3].T, [0.0, 0.0, 1.0]))
        vlog.p(f"Image `{index + 1}`:")
        vlog.code(np.array2string(transform, precision=8))
        transforms.append(transform)

    try:
        _log_alignment(vlog, title, stitching_images, transforms, matches)
    except (cv.error, MemoryError) as error:
        vlog.err(f"Could not render the full-resolution alignment: {error}")

    vlog.end_timer()


def _log_alignment(
        vlog: MdLogger,
        title: str,
        stitching_images: StitchingImages,
        transforms: Sequence[FloatArray],
        matches: Sequence[PairwiseMatch],
) -> None:
    """Overlay original scans using global rigid transforms and one shared canvas offset."""
    corners = []
    with np.errstate(over="ignore", invalid="ignore"):
        for image, transform in zip(stitching_images.scan_images, transforms):
            height, width = image.shape[:2]
            points = np.array(((0, 0), (width, 0), (width, height), (0, height)), dtype=np.float64)
            corners.append(points @ transform[:2, :2].T + transform[:2, 2])
        corners = np.vstack(corners)
        minimum = np.floor(corners.min(axis=0))
        size = np.ceil(corners.max(axis=0)) - minimum
    if not np.isfinite(corners).all() or not np.isfinite(size).all() or np.any(size <= 0):
        vlog.err("Cannot render full-resolution alignment: non-finite or empty transformed image bounds.")
        return

    canvas_width, canvas_height = int(size[0]), int(size[1])
    display_affines = []
    image_sum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    coverage = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    for image, transform in zip(stitching_images.scan_images, transforms):
        display_affine = transform[:2].copy()
        display_affine[:, 2] -= minimum
        display_affines.append(display_affine)
        warped_image = cv.warpAffine(image, display_affine, (canvas_width, canvas_height))
        warped_mask = cv.warpAffine(
            np.ones(image.shape[:2], dtype=np.float32), display_affine, (canvas_width, canvas_height),
        )
        image_sum += warped_image
        coverage += warped_mask
        del warped_image, warped_mask

    np.divide(image_sum, coverage[:, :, None], out=image_sum, where=coverage[:, :, None] > 0)
    image_sum[coverage == 0] = 255
    np.rint(image_sum, out=image_sum)
    np.clip(image_sum, 0, 255, out=image_sum)
    overlay = image_sum.astype(np.uint8)
    del image_sum, coverage

    vlog.h5("Global alignment")
    vlog.p(
        f"Full-resolution canvas: `{canvas_width} x {canvas_height}` px; "
        f"shared canvas offset: `({-minimum[0]:.0f}, {-minimum[1]:.0f})` px. "
        "Overlapping images are averaged; uncovered areas are white."
    )
    vlog.img(overlay, alt=f"{title}: global rigid overlay", style="max-height: 500px;")

    for pair in matches:
        if not pair.accepted:
            continue
        affine_i = display_affines[pair.image_idx_i]
        affine_j = display_affines[pair.image_idx_j]
        with np.errstate(over="ignore", invalid="ignore"):
            points_i = pair.keypoints_i @ affine_i[:, :2].T + affine_i[:, 2]
            points_j = pair.keypoints_j @ affine_j[:, :2].T + affine_j[:, 2]
        for point_i, point_j in zip(points_i, points_j):
            if not np.isfinite((point_i, point_j)).all() or np.max(np.abs((point_i, point_j))) > 2 ** 30:
                continue
            cv.line(
                overlay,
                (round(float(point_i[0])), round(float(point_i[1]))),
                (round(float(point_j[0])), round(float(point_j[1]))),
                (0, 255, 0), thickness=1, lineType=cv.LINE_AA,
            )

    vlog.br()
    vlog.img(
        overlay, link_only=True, alt=f"{title}: residual connection lines", style="max-height: 500px;",
    )


def check_connectivity(matches: Sequence[PairwiseMatch], image_count: int) -> None:
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
