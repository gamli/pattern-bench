from pathlib import Path
from typing import Sequence

import cv2 as cv

from cli.commands.utils import print_info, start_info_timer, print_peak_memory
from utils.md_logger import MdLogger

from ._verbose_logging import _vlog_end_timer, _vlog_start_timer
from .shared import ImageArray, StitchingConfig, StitchingImages


def detect_features(
        stitching_images: StitchingImages,
        config: StitchingConfig,
) -> tuple[cv.detail.ImageFeatures, ...]:
    """Compute uncapped SIFT keypoints and descriptors on full-size grayscale images.

    SIFT's internal scale space is part of the detector. No input image is
    resized, and returned keypoints use the original pixel coordinates.
    """

    print_info("Detecting Features")
    info_timer = start_info_timer()

    vlog = config.vlog

    _vlog_feature_detection_start(vlog, stitching_images.count)

    _vlog_start_timer(vlog, "cv.SIFT.create(...)")
    finder = cv.SIFT.create()
    _vlog_end_timer(vlog)

    features = []

    for image_idx, [file, image] in enumerate(stitching_images.files_and_images):

        print_info(f"Computing Image Features for: {file.name}")

        _vlog_image_detection_start(vlog, file)

        _vlog_start_timer(vlog, "config.apply_fixed_seed(...)")
        config.apply_fixed_seed()
        _vlog_end_timer(vlog)

        _vlog_start_timer(vlog, "image_features = cv.detail.computeImageFeatures2(...)")
        image_features = cv.detail.computeImageFeatures2(finder, image)
        _vlog_end_timer(vlog)

        image_features.img_idx = image_idx
        features.append(image_features)

        _vlog_start_timer(vlog, "image_features.getKeypoints(...)")
        keypoints = image_features.getKeypoints()
        _vlog_end_timer(vlog)

        feature_count = len(keypoints)

        _vlog_detected_features(vlog, file, image, keypoints)

        if feature_count < config.num_matches_thresh1:
            _vlog_insufficient_features(vlog, file, feature_count)
            raise RuntimeError(f"Image {file} has too few SIFT keypoints: {feature_count}.")

        print_peak_memory("After Image")
        info_timer.print(f"\tDone")

    _vlog_feature_detection_finish(vlog)

    return tuple(features)

#region vlog

def _vlog_feature_detection_start(vlog: MdLogger | None, image_count: int) -> None:
    if not vlog:
        return

    vlog.h3("Feature Detection")
    vlog.start_timer("Feature Detection (total)")
    vlog.log_elapsed("Feature Detection starting at")
    vlog.p(f"Detecting features on `{image_count}` images using SIFT")


def _vlog_image_detection_start(vlog: MdLogger | None, file: Path) -> None:
    if vlog:
        vlog.p(f"Detecting features on image `{file.name}`")


def _vlog_detected_features(
        vlog: MdLogger | None,
        file: Path,
        image: ImageArray,
        keypoints: Sequence[cv.KeyPoint],
) -> None:
    if not vlog:
        return

    vlog.p(f"Found `{len(keypoints)}` features.")
    vlog.start_timer("Logging image with keypoints")
    vlog.img(
        cv.drawKeypoints(
            image,
            keypoints,
            image.copy(),
            color=(0, 255, 0),
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS | cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
        ),
        alt=f"SIFT keypoints in {file.name}",
        style="max-height: 500px;",
    )
    vlog.end_timer()


def _vlog_insufficient_features(vlog: MdLogger | None, file: Path, feature_count: int) -> None:
    if vlog:
        vlog.err(f"Image `{file}` has too few SIFT keypoints: `{feature_count}`.")


def _vlog_feature_detection_finish(vlog: MdLogger | None) -> None:
    if not vlog:
        return

    vlog.end_timer()
    vlog.log_elapsed("Feature Detection finished at")

#endregion
