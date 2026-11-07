from dataclasses import dataclass
from typing import Sequence, cast

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from cli.commands.utils import print_info, start_info_timer
from .shared import ImageArray, FloatArray, StitchingImages


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
) -> CompositionResult:
    """Warp originals at full resolution and copy pixels across hard DP seams.

    The output retains the reference coordinate frame's orientation. Bounds
    are translated onto a positive canvas; no additional rotation is applied.
    COLOR_GRAD dynamic-programming seams select one source per covered pixel.
    There is no feathering, color compensation, or blending between sources.
    Color uses INTER_LINEAR with replicated borders to avoid dark edge samples;
    independent INTER_NEAREST support masks exclude pixels outside the scans.
    """

    print_info("Composing images")
    info_timer = start_info_timer()

    if not stitching_images.count or len(image_transforms) != stitching_images.count:
        raise ValueError("Provide at least one image and one affine transform per image.")

    source_transforms: list[FloatArray] = []
    all_corners: list[NDArray[np.float64]] = []

    print_info("Initializing transforms")

    for i in range(len(image_transforms)):
        transform = np.asarray(image_transforms[i], dtype=np.float64)
        if (transform.shape != (3, 3)
                or not np.isfinite(transform).all()
                or not np.array_equal(transform[2], [0.0, 0.0, 1.0])):
            raise ValueError("Transforms must be finite homogeneous affine 3x3 matrices.")
        if abs(np.linalg.det(transform[:2, :2])) < 1e-12:
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
        raise ValueError("Transformed image bounds must be finite and nonempty.")
    width, height = map(int, size)

    coverage = np.zeros((height, width), dtype=np.min_scalar_type(stitching_images.count))
    canvas_transforms = []
    warped_images = []
    supports = []
    canvas_corners = []
    covered_counts = []

    info_timer.print(f"\tDone")

    print_info("Warping images")

    for i, (image, source_transform) in enumerate(zip(stitching_images.scan_images, source_transforms)):
        transform = source_transform.copy()
        transform[:2, 2] -= lower
        canvas_transforms.append(transform)

        x, y = map(int, np.floor(all_corners[i].min(axis=0)) - lower)
        right, bottom = map(int, np.ceil(all_corners[i].max(axis=0)) - lower)
        local_transform = transform[:2].copy()
        local_transform[:, 2] -= (x, y)
        local_size = (right - x, bottom - y)
        warped = cv.warpAffine(
            image, local_transform, local_size,
            flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE,
        )
        support = cv.warpAffine(
            np.full(image.shape[:2], 255, dtype=np.uint8), local_transform, local_size,
            flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0,
        )
        valid = support != 0
        warped[~valid] = 0
        coverage[y:bottom, x:right] += valid
        covered_counts.append(int(np.count_nonzero(valid)))
        warped_images.append(warped)
        supports.append(support)
        canvas_corners.append((x, y))

    info_timer.print(f"\tDone")

    print_info("Finding seams")
    seam_masks = cv.detail.DpSeamFinder("COLOR_GRAD").find(
        warped_images,
        canvas_corners,
        [cv.UMat(support) for support in supports],
    )
    info_timer.print(f"\tDone")

    print_info("Stitching with seams")

    owners = np.zeros_like(coverage)
    conflicts = 0
    for i, (seam_mask, support, (x, y)) in enumerate(zip(seam_masks, supports, canvas_corners)):
        seam_mask = seam_mask.get()
        region = owners[y:y + support.shape[0], x:x + support.shape[1]]
        candidate = (seam_mask != 0) & (support != 0)
        conflicts += int(np.count_nonzero(candidate & (region != 0)))
        np.copyto(region, i + 1, where=candidate & (region == 0))
    del seam_masks, seam_mask, candidate

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
    output_with_seam_areas = cast(ImageArray, cv.addWeighted(output, 0.6, seam_areas, 0.4, 0))

    output_dpis = [
        dpi * np.sqrt(abs(np.linalg.det(transform[:2, :2])))
        for dpi, transform in zip(stitching_images.scan_dpis, image_transforms)
        if dpi is not None
    ]
    output_dpi = float(np.median(output_dpis)) if output_dpis else None

    info_timer.print(f"\tDone")

    return CompositionResult(
        image=output,
        image_with_seams=output_with_seams,
        image_with_seam_areas=output_with_seam_areas,
        dpi=output_dpi,
        coverage=coverage,
        transforms=tuple(canvas_transforms),
    )
