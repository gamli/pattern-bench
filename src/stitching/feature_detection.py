import cv2 as cv

from src.stitching.shared import StitchingConfig, StitchingImages


def detect_features(
        stitching_images: StitchingImages,
        config: StitchingConfig,
) -> tuple[cv.detail.ImageFeatures, ...]:
    """Compute uncapped SIFT keypoints and descriptors on full-size grayscale images.

    SIFT's internal scale space is part of the detector. No input image is
    resized, and returned keypoints use the original pixel coordinates.
    """

    vlog = config.vlog

    if vlog:
        vlog.h3("Feature Detection")
        vlog.start_timer(f"Feature Detection (total)")
        vlog.log_elapsed("Feature Detection starting at")
        vlog.p(f"Detecting features on `{stitching_images.count}` images using SIFT")

    if vlog:
        vlog.start_timer(f"cv.SIFT.create(...)")
    finder = cv.SIFT.create()
    if vlog:
        vlog.end_timer()

    features = []

    for image_idx, [file, image] in enumerate(stitching_images.files_and_images):

        if vlog:
            vlog.p(f"Detecting features on image `{file.name}`")

        if vlog:
            vlog.start_timer(f"config.apply_fixed_seed(...)")
        config.apply_fixed_seed()
        if vlog:
            vlog.end_timer()

        if vlog:
            vlog.start_timer(f"image_features = cv.detail.computeImageFeatures2(...)")
        image_features = cv.detail.computeImageFeatures2(finder, image)
        if vlog:
            vlog.end_timer()

        image_features.img_idx = image_idx
        features.append(image_features)

        if vlog:
            vlog.start_timer(f"image_features.getKeypoints(...)")
        keypoints = image_features.getKeypoints()
        if vlog:
            vlog.end_timer()

        feature_count = len(keypoints)

        if vlog:
            vlog.p(f"Found `{feature_count}` features.")
            vlog.start_timer(f"Logging image with keypoints")
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

        if feature_count < config.num_matches_thresh1:
            if vlog:
                vlog.err(f"Image `{file}` has too few SIFT keypoints: `{feature_count}`.")
            raise RuntimeError(f"Image {file} has too few SIFT keypoints: {feature_count}.")

    if vlog:
        vlog.end_timer()
        vlog.log_elapsed("Feature Detection finished at")

    return tuple(features)
