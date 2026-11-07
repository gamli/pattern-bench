from typing import Sequence, cast

import cv2 as cv
import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from cli.commands.utils import print_info, start_info_timer
from utils.md_logger import MdLogger

from ._rigid_geometry import fit_rigid_transform
from ._verbose_logging import _vlog_end_timer, _vlog_error, _vlog_start_timer
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

    vlog = config.vlog

    _vlog_estimation_start(vlog, stitching_images.count, config)

    try:
        _vlog_start_timer(vlog, "check_connectivity(...)")
        print_info("Checking Connectivity")
        _check_connectivity(pairwise_matches, stitching_images.count)
        info_timer.print(f"\tDone")
        _vlog_end_timer(vlog)
    except (ValueError, RuntimeError) as error:
        _vlog_error(vlog, error)
        raise

    if stitching_images.count == 1:
        _vlog_single_image_estimation(vlog)
        return (np.eye(3, dtype=np.float64),)

    for pair in pairwise_matches:
        if not pair.accepted:
            continue
        p, q = pair.keypoints_i, pair.keypoints_j
        if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
            _vlog_invalid_pair(vlog, pair, "invalid matched point array shapes")
            raise ValueError("Accepted pairs need point arrays with shape (N, 2), N >= 2.")
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            _vlog_invalid_pair(vlog, pair, "matched coordinates must be finite")
            raise ValueError("Matched coordinates must be finite.")

    print_info("Computing pixel pyramids")
    pixel_pyramids = None
    if config.pixel_refinement:
        _vlog_start_timer(vlog, "Preparing pixel contrast pyramids")
        pixel_pyramids = prepare_pixel_images(stitching_images)
        _vlog_end_timer(vlog)
        pairwise_matches = validate_pixel_pairs(pairwise_matches, pixel_pyramids, config)
        try:
            _check_connectivity(pairwise_matches, stitching_images.count)
        except RuntimeError as error:
            message = f"Pixel validation removed unreliable pairs. {error}"
            _vlog_error(vlog, message)
            _vlog_estimation_finish(vlog)
            raise RuntimeError(message) from error
    info_timer.print(f"\tDone")

    _vlog_start_timer(vlog, "coefficient/target initialization")

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

    _vlog_end_timer(vlog)

    _vlog_estimation_system(vlog, pairwise_matches, coefficient_blocks, coefficients, targets)

    _vlog_start_timer(vlog, "Kabsch initialization")
    print_info("Computing Initial Kabsch Solution")
    solution = _initial_rigid_solution(pairwise_matches, stitching_images.count)
    info_timer.print(f"\tDone")
    _vlog_end_timer(vlog)

    _vlog_start_timer(vlog, "Initial Levenberg-Marquardt")
    print_info("Computing Initial Levenberg-Marquardt Solution")
    solution, rank = _solve_rigid(coefficients, targets, solution, vlog=vlog)
    info_timer.print(f"\tDone")
    _vlog_end_timer(vlog)

    _vlog_estimation_step(
        vlog, "Initial least squares", solution, coefficients, targets,
        pairwise_matches, stitching_images, config.huber_delta, rank=rank,
    )

    print_info("Running IRLS iterations")
    for iteration in range(config.irls_iterations):

        _vlog_start_timer(vlog, "IRLS iteration {} / {}", iteration + 1, config.irls_iterations)

        _vlog_start_timer(vlog, "np.linalg.norm(...)")
        errors = np.linalg.norm(coefficients @ solution - targets, axis=1)
        _vlog_end_timer(vlog)

        _vlog_start_timer(vlog, "np.sqrt(...)")
        # sqrt(weight) multiplies the least-squares rows, not weight itself.
        weights = np.sqrt(np.minimum(1.0, config.huber_delta / np.maximum(errors, 1e-12)))
        _vlog_end_timer(vlog)

        _vlog_start_timer(vlog, "Weighted Levenberg-Marquardt")
        solution, _ = _solve_rigid(coefficients, targets, solution, weights, vlog)
        _vlog_end_timer(vlog)

        _vlog_estimation_step(
            vlog, "IRLS iteration {} / {}", cast(FloatArray, solution), coefficients, targets,
            pairwise_matches, stitching_images, config.huber_delta, weights,
            title_args=(iteration + 1, config.irls_iterations),
        )
        _vlog_end_timer(vlog)
    info_timer.print(f"\tDone")

    transforms = [np.eye(3, dtype=np.float64)]
    for index in range(stitching_images.count - 1):
        transform = np.vstack((solution[3 * index:3 * index + 3].T, [0.0, 0.0, 1.0]))
        if not np.isfinite(transform).all() or abs(np.linalg.det(transform[:2, :2])) < 1e-12:
            message = f"Degenerate global rigid transform for image {index + 1}."
            _vlog_error(vlog, message)
            raise RuntimeError(message)
        transforms.append(transform)

    if pixel_pyramids is not None:
        print_info("Running pixel based refinement")
        transforms = list(refine_global_pixels(transforms, pairwise_matches, pixel_pyramids, config))
        try:
            _vlog_alignment(vlog, "Pixel refinement", stitching_images, transforms, pairwise_matches)
        except (cv.error, MemoryError) as error:
            _vlog_error(vlog, f"Could not render the pixel-refined alignment: {error}")
        info_timer.print(f"\tDone")

    _vlog_estimation_result(vlog, transforms)

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
    _vlog_solver_result(vlog, result)
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
        message = f"Rigid registration failed: {result.message}"
        _vlog_error(vlog, message)
        raise RuntimeError(message)
    rank = int(np.linalg.matrix_rank(result.jac))
    if rank != len(result.x):
        message = f"Rigid registration is underdetermined: rank {rank}, expected {len(result.x)}."
        _vlog_error(vlog, message)
        raise RuntimeError(message)
    return make_solution(result.x), rank

#region vlog

def _vlog_estimation_start(vlog: MdLogger | None, image_count: int, config: StitchingConfig) -> None:
    if not vlog:
        return

    vlog.h3("Global Transformation Estimation")
    vlog.start_timer("Global Transformation Estimation (total)")
    vlog.log_elapsed("Global Transformation Estimation starting at")
    vlog.p(f"Registering `{image_count}` images jointly; image `0` is fixed to identity.")
    vlog.table(
        ("Parameter", "Value"),
        (
            ("irls_iterations", config.irls_iterations),
            ("huber_delta (px)", config.huber_delta),
            ("pixel_refinement", config.pixel_refinement),
            ("pixel_min_correlation", config.pixel_min_correlation),
            ("pixel_max_correction (px)", config.pixel_max_correction),
            ("pixel_refinement_iterations", config.pixel_refinement_iterations),
            ("pixel_max_samples (per pair)", config.pixel_max_samples),
            ("solver", "SciPy least_squares / Levenberg-Marquardt"),
            ("initialization", "Pairwise Kabsch / spanning tree"),
        ),
    )


def _vlog_single_image_estimation(vlog: MdLogger | None) -> None:
    if not vlog:
        return

    vlog.p("Only one image was provided; returning identity without least squares or IRLS.")
    vlog.code(np.array2string(np.eye(3, dtype=np.float64), precision=8))
    _vlog_estimation_finish(vlog)


def _vlog_invalid_pair(vlog: MdLogger | None, pair: PairwiseMatch, reason: str) -> None:
    if not vlog:
        return

    vlog.err(f"Pair {pair.image_idx_i} / {pair.image_idx_j}: {reason}.")
    _vlog_estimation_finish(vlog)


def _vlog_estimation_system(
        vlog: MdLogger | None,
        pairwise_matches: Sequence[PairwiseMatch],
        coefficient_blocks: Sequence[FloatArray],
        coefficients: FloatArray,
        targets: FloatArray,
) -> None:
    if not vlog:
        return

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


def _vlog_estimation_result(vlog: MdLogger | None, transforms: Sequence[FloatArray]) -> None:
    if not vlog:
        return

    vlog.h4("Global transformation estimation result")
    vlog.p(f"Returning `{len(transforms)}` global transforms mapping each image to reference image `0`.")
    for index, transform in enumerate(transforms):
        vlog.p(f"Final transform for image `{index}`:")
        vlog.code(np.array2string(transform, precision=8))
    _vlog_estimation_finish(vlog)


def _vlog_estimation_finish(vlog: MdLogger | None) -> None:
    if not vlog:
        return

    vlog.end_timer()
    vlog.log_elapsed("Global Transformation Estimation finished at")


def _vlog_solver_result(vlog: MdLogger | None, result: OptimizeResult) -> None:
    if not vlog:
        return

    vlog.table(
        ("Solver metric", "Value"),
        (
            ("Termination", result.message),
            ("Function evaluations", result.nfev),
            ("Jacobian evaluations", result.njev),
            ("First-order optimality", f"{result.optimality:.6g}"),
        ),
    )


def _vlog_estimation_step(
        vlog: MdLogger | None,
        title: str,
        solution: FloatArray,
        coefficients: FloatArray,
        targets: FloatArray,
        matches: Sequence[PairwiseMatch],
        stitching_images: StitchingImages,
        huber_delta: float,
        row_weights: FloatArray | None = None,
        *,
        rank: int | None = None,
        title_args: tuple[object, ...] = (),
) -> None:
    """Report the new solution and the optional row weights used for this solve."""
    if not vlog:
        return

    title = title.format(*title_args) if title_args else title
    errors = np.linalg.norm(coefficients @ solution - targets, axis=1)
    if rank is not None:
        vlog.p(f"The rigid-fit Jacobian has full column rank: `{rank}`.")

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
        _vlog_alignment(vlog, title, stitching_images, transforms, matches)
    except (cv.error, MemoryError) as error:
        vlog.err(f"Could not render the full-resolution alignment: {error}")

    vlog.end_timer()


def _vlog_alignment(
        vlog: MdLogger | None,
        title: str,
        stitching_images: StitchingImages,
        transforms: Sequence[FloatArray],
        matches: Sequence[PairwiseMatch],
) -> None:
    """Overlay original scans using global rigid transforms and one shared canvas offset."""
    if not vlog:
        return

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

#endregion
