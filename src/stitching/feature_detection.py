import cv2 as cv

from cli.commands.utils import print_info, start_info_timer, print_peak_memory
from .shared import StitchingConfig, StitchingImages


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

    finder = cv.SIFT.create()

    features = []

    for image_idx, [file, image] in enumerate(stitching_images.files_and_images):

        print_info(f"Computing Image Features for: {file.name}")

        config.apply_fixed_seed()

        image_features = cv.detail.computeImageFeatures2(finder, image)

        image_features.img_idx = image_idx
        features.append(image_features)

        keypoints = image_features.getKeypoints()

        feature_count = len(keypoints)

        if feature_count < config.num_matches_thresh1:
            raise RuntimeError(f"Image {file} has too few SIFT keypoints: {feature_count}.")

        print_peak_memory("After Image")
        info_timer.print(f"\tDone")

    return tuple(features)
