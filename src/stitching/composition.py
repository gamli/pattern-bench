from dataclasses import dataclass
from typing import Sequence

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from src.stitching.shared import ImageArray, FloatArray, StitchingImages, StitchingConfig


@dataclass(frozen=True)
class CompositionResult:
    """BGR output, source coverage counts, and source-to-output affine transforms.

    coverage[y, x] counts source images covering that pixel before seam selection.
    The result takes each covered pixel from exactly one warped source image.
    Uncovered pixels are black. Image data is not reduced in resolution.
    """

    image: ImageArray
    image_with_seams: ImageArray
    image_with_seam_areas: ImageArray
    dpi: float | None
    coverage: NDArray[np.unsignedinteger]
    transforms: tuple[FloatArray, ...]


def compose_images(
        image_transforms: Sequence[FloatArray],
        stitching_images: StitchingImages,
        config: StitchingConfig,
) -> CompositionResult:
    """Warp originals at full resolution and copy pixels across hard DP seams.

    The output retains the reference coordinate frame's orientation. Bounds
    are translated onto a positive canvas; no additional rotation is applied.
    COLOR_GRAD dynamic-programming seams select one source per covered pixel.
    There is no feathering, color compensation, or blending between sources.
    Color uses INTER_LINEAR with replicated borders to avoid dark edge samples;
    independent INTER_NEAREST support masks exclude pixels outside the scans.
    """

    vlog = config.vlog
    if vlog:
        vlog.h3("Image Composition")
        vlog.start_timer(f"Image Composition (total)")
        vlog.log_elapsed("Image Composition starting at")
        vlog.p(f"Composing `{stitching_images.count}` images at full resolution using hard seams.")
        vlog.p("Seam finder: OpenCV `DpSeamFinder`, cost `COLOR_GRAD`; each output pixel uses one warped source.")

    if not stitching_images.count or len(image_transforms) != stitching_images.count:
        if vlog:
            vlog.err("Provide at least one image and one affine transform per image.")
            vlog.end_timer()
            vlog.log_elapsed("Image Composition finished at")
        raise ValueError("Provide at least one image and one affine transform per image.")

    if vlog:
        vlog.start_timer(f"Checking transforms")

    source_transforms: list[FloatArray] = []
    all_corners: list[NDArray[np.float64]] = []

    for i in range(len(image_transforms)):
        transform = np.asarray(image_transforms[i], dtype=np.float64)
        if (transform.shape != (3, 3) or not np.isfinite(transform).all()
                or not np.array_equal(transform[2], [0.0, 0.0, 1.0])):
            if vlog:
                vlog.err(f"Image {i}: transform must be a finite homogeneous affine 3x3 matrix.")
                vlog.end_timer()
                vlog.log_elapsed("Image Composition finished at")
            raise ValueError("Transforms must be finite homogeneous affine 3x3 matrices.")
        if abs(np.linalg.det(transform[:2, :2])) < 1e-12:
            if vlog:
                vlog.err(f"Image {i}: affine transform must be invertible.")
                vlog.end_timer()
                vlog.log_elapsed("Image Composition finished at")
            raise ValueError("Affine transforms must be invertible.")
        source_transforms.append(transform)
        height, width = stitching_images.scan_images[i].shape[:2]
        corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
        with np.errstate(over="ignore", invalid="ignore"):
            all_corners.append(corners @ transform[:2, :2].T + transform[:2, 2])

    corners = np.vstack(all_corners)
    lower = np.floor(corners.min(axis=0))
    upper = np.ceil(corners.max(axis=0))
    with np.errstate(over="ignore", invalid="ignore"):
        size = upper - lower
    if not np.isfinite(size).all() or np.any(size <= 0):
        if vlog:
            vlog.err("Transformed image bounds must be finite and nonempty.")
            vlog.end_timer()
            vlog.log_elapsed("Image Composition finished at")
        raise ValueError("Transformed image bounds must be finite and nonempty.")
    width, height = map(int, size)

    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.p(f"Output canvas: `{width} x {height}` px; shared offset: `({-lower[0]:.0f}, {-lower[1]:.0f})` px.")
        vlog.h4("Warping source images")

    coverage = np.zeros((height, width), dtype=np.min_scalar_type(stitching_images.count))
    canvas_transforms = []
    warped_images = []
    supports = []
    canvas_corners = []
    covered_counts = []

    if vlog:
        vlog.start_timer(f"Warping all images")

    for i, (image, source_transform) in enumerate(zip(stitching_images.scan_images, source_transforms)):

        if vlog:
            vlog.start_timer(f"Warping image {i + 1} / {stitching_images.count}")

        transform = source_transform.copy()
        transform[:2, 2] -= lower
        canvas_transforms.append(transform)

        x, y = map(int, np.floor(all_corners[i].min(axis=0)) - lower)
        right, bottom = map(int, np.ceil(all_corners[i].max(axis=0)) - lower)
        local_transform = transform[:2].copy()
        local_transform[:, 2] -= (x, y)
        local_size = (right - x, bottom - y)
        try:
            warped = cv.warpAffine(
                image, local_transform, local_size,
                flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE,
            )
            support = cv.warpAffine(
                np.full(image.shape[:2], 255, dtype=np.uint8), local_transform, local_size,
                flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0,
            )
        except cv.error as error:
            if vlog:
                vlog.err(f"Affine warping failed for image {i}: {error}")
                vlog.end_timer()
                vlog.log_elapsed("Image Composition finished at")
            raise
        valid = support != 0
        warped[~valid] = 0
        coverage[y:bottom, x:right] += valid
        covered_counts.append(int(np.count_nonzero(valid)))
        warped_images.append(warped)
        supports.append(support)
        canvas_corners.append((x, y))

        if vlog:
            vlog.p(
                f"Image `{i}` (`{stitching_images.scan_files[i]}`): "
                f"warped to `{local_size[0]} x {local_size[1]}` px at `({x}, {y})`; "
                f"covers `{covered_counts[-1]}` pixels."
            )
            vlog.code(np.array2string(transform, precision=8))

        if vlog:
            vlog.end_timer()

    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.table(
            ("Coverage", "Pixels"),
            (
                ("Covered", np.count_nonzero(coverage)),
                ("Overlapping (at least two images)", np.count_nonzero(coverage > 1)),
                ("Uncovered", np.count_nonzero(coverage == 0)),
            ),
        )
        vlog.h4("Averaged overlay")
        if vlog:
            vlog.start_timer(f"Logging Averaged overlay")
        seam_areas = _average_overlay(warped_images, canvas_corners, coverage)
        vlog.img(seam_areas, alt="All warped scans averaged in overlaps", style="max-height: 500px;")
        del seam_areas
        if vlog:
            vlog.end_timer()

    if vlog:
        vlog.start_timer(f"Full-resolution `COLOR_GRAD` seaming")

    try:
        seam_masks = cv.detail.DpSeamFinder("COLOR_GRAD").find(
            warped_images,
            canvas_corners,
            [cv.UMat(support) for support in supports],
        )
    except cv.error as error:
        if vlog:
            vlog.err(f"Dynamic-programming seam estimation failed: {error}")
            vlog.end_timer()
            vlog.log_elapsed("Image Composition finished at")
        raise

    owners = np.zeros_like(coverage)
    conflicts = 0
    for i, (seam_mask, support, (x, y)) in enumerate(zip(seam_masks, supports, canvas_corners)):
        seam_mask = seam_mask.get()
        region = owners[y:y + support.shape[0], x:x + support.shape[1]]
        candidate = (seam_mask != 0) & (support != 0)
        conflicts += int(np.count_nonzero(candidate & (region != 0)))
        np.copyto(region, i + 1, where=candidate & (region == 0))
    del seam_masks, seam_mask, candidate

    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.start_timer(f"Resolve ties")

    # Coincident scans can retain overlapping seam masks. Resolve ties by image
    # index, and restore any seam holes from valid support without mixing colors.
    restored = int(np.count_nonzero((coverage != 0) & (owners == 0)))
    if restored:
        for i, (support, (x, y)) in enumerate(zip(supports, canvas_corners)):
            region = owners[y:y + support.shape[0], x:x + support.shape[1]]
            np.copyto(region, i + 1, where=(region == 0) & (support != 0))

    output = np.zeros((height, width, 3), dtype=np.uint8)
    selected_counts = []
    for i, (warped, (x, y)) in enumerate(zip(warped_images, canvas_corners)):
        region = output[y:y + warped.shape[0], x:x + warped.shape[1]]
        selected = owners[y:y + warped.shape[0], x:x + warped.shape[1]] == i + 1
        np.copyto(region, warped, where=selected[:, :, None])
        selected_counts.append(int(np.count_nonzero(selected)))
    del warped_images, supports

    seams = np.zeros(owners.shape, dtype=np.uint8)
    seams[:, 1:] |= (owners[:, 1:] != owners[:, :-1]) & (owners[:, 1:] != 0) & (owners[:, :-1] != 0)
    seams[1:] |= (owners[1:] != owners[:-1]) & (owners[1:] != 0) & (owners[:-1] != 0)
    seams = cv.dilate(seams, np.ones((3, 3), dtype=np.uint8))
    output_with_seams = output.copy()
    output_with_seams[seams != 0] = (0, 255, 0)

    seam_area_colors = np.array([
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ], dtype=np.uint8)
    seam_area_mask = owners != 0
    seam_areas = output.copy()
    seam_areas[seam_area_mask] = seam_area_colors[(owners[seam_area_mask] - 1) % len(seam_area_colors)]
    output_with_seam_areas = cv.addWeighted(output, 0.6, seam_areas, 0.4, 0)

    if vlog:
        vlog.end_timer()

    if vlog:
        vlog.p(
            f"Conflicting seam claims resolved by image index: `{conflicts}`; seam holes restored: `{restored}` pixels.")
        vlog.table(
            ("Image", "Covered pixels before seams", "Selected output pixels"),
            tuple((index, covered, selected) for index, (covered, selected) in
                  enumerate(zip(covered_counts, selected_counts))),
        )
        vlog.h4("Hard-seam result")
        vlog.p("Every covered output pixel is copied from one warped image; uncovered pixels remain black.")
        vlog.img(output, alt="Stitched scans with hard seams", style="max-height: 500px;")
        vlog.img(output_with_seams, link_only=True, alt="Stitching Seams", style="max-height: 500px;")
        vlog.img(output_with_seam_areas, link_only=True, alt="Stitching Seam Areas", style="max-height: 500px;")
        vlog.end_timer()
        vlog.log_elapsed("Image Composition finished at")

    output_dpis = [
        dpi * np.sqrt(abs(np.linalg.det(transform[:2, :2])))
        for dpi, transform in zip(stitching_images.scan_dpis, image_transforms)
        if dpi is not None
    ]
    output_dpi = float(np.median(output_dpis)) if output_dpis else None

    return CompositionResult(
        image=output,
        image_with_seams=output_with_seams,
        image_with_seam_areas=output_with_seam_areas,
        dpi=output_dpi,
        coverage=coverage,
        transforms=tuple(canvas_transforms),
    )


def _average_overlay(
        warped_images: Sequence[ImageArray],
        corners: Sequence[tuple[int, int]],
        coverage: NDArray[np.unsignedinteger],
) -> ImageArray:
    """Average original support in strips, without a full floating-point canvas."""
    height, width = coverage.shape
    overlay = np.empty((height, width, 3), dtype=np.uint8)
    for row in range(0, height, 256):
        bottom = min(row + 256, height)
        accumulated = np.zeros((bottom - row, width, 3), dtype=np.float32)
        for image, (x, y) in zip(warped_images, corners):
            top = max(row, y)
            end = min(bottom, y + image.shape[0])
            if top < end:
                accumulated[top - row:end - row, x:x + image.shape[1]] += image[top - y:end - y]
        np.divide(accumulated, np.maximum(coverage[row:bottom, :, None], 1), out=accumulated)
        np.rint(accumulated, out=accumulated)
        np.clip(accumulated, 0, 255, out=accumulated)
        overlay[row:bottom] = accumulated.astype(np.uint8)
    return overlay
