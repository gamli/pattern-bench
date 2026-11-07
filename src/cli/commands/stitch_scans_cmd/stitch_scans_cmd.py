import argparse
import shutil
from pathlib import Path
from typing import Protocol

from scanner.calibration import apply_scanner_calibration, ScannerCalibration
from stitching.composition import compose_images
from stitching.feature_detection import detect_features
from stitching.feature_matching import match_features
from stitching.shared import StitchingConfig
from stitching.transformation_estimation import estimate_global_transforms
from ..utils import save_output_image, load_scan_images, print_info, start_info_timer, print_peak_memory


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "stitch-scans",
        help="Stitches a number of individual scans into a single large scan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "scan_files",
        metavar="SCAN_FILES",
        nargs="+",
        help="The scan files or file patterns to stitch.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        metavar="OUTPUT_DIR",
        type=Path,
        default=Path(),
        help="Path to save the output (stitched scan image and verbose) to.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "-f", "--use-fixed-seed",
        action="store_true",
        help="Reset the RNG to a fixed seed before each random operation.",
    )
    parser.add_argument(
        "--no-pixel-refinement",
        action="store_true",
        help="Disable pixel validation of image pairs and final pixel refinement.",
    )

    parser.set_defaults(handler=_run)


class StitchScansArgs(Protocol):
    scan_files: list[str]
    output_dir: Path
    verbose: bool
    use_fixed_seed: bool
    no_pixel_refinement: bool


def _run(args: StitchScansArgs) -> None:
    print_info("Running stitch-scans")

    info_timer = start_info_timer()

    print_info("Loading images")
    stitching_images = load_scan_images(args.scan_files)
    info_timer.print("\tDone")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_image_file = args.output_dir / "stitched_scans.jpg"
    output_image_file.unlink(missing_ok=True)

    output_image_with_seams_file = args.output_dir / "stitched_scans_with_seams.jpg"
    output_image_with_seams_file.unlink(missing_ok=True)

    output_image_with_seam_areas_file = args.output_dir / "stitched_scans_with_seam_areas.jpg"
    output_image_with_seam_areas_file.unlink(missing_ok=True)

    verbose_dir = args.output_dir / "verbose"
    if verbose_dir.exists():
        shutil.rmtree(verbose_dir)
    if args.verbose:
        verbose_dir.mkdir()

    config = StitchingConfig(
        use_fixed_seed=args.use_fixed_seed,
        pixel_refinement=not args.no_pixel_refinement,
    )
    config.configure_opencv()

    scanner_calibration = ScannerCalibration()

    print_peak_memory("Initial")

    print_info("1. Calibration")
    corrected_stitching_images = apply_scanner_calibration(stitching_images, scanner_calibration)
    print_peak_memory("After Calibration")
    info_timer.print("\tDone")

    print_info("2. Detect Features")
    features = detect_features(corrected_stitching_images, config)
    print_peak_memory("After Detect Features")
    info_timer.print("\tDone")

    print_info("3. Match Features")
    pairwise_matches = match_features(features, corrected_stitching_images, config)
    print_peak_memory("After ")
    info_timer.print("\tDone")

    print_info("4. Transforms")
    image_transforms = estimate_global_transforms(pairwise_matches, corrected_stitching_images, config)
    print_peak_memory("After Match Features")
    info_timer.print("\tDone")

    print_info("5. Compose")
    composed = compose_images(image_transforms, corrected_stitching_images)
    print_peak_memory("After Compose")
    info_timer.print("\tDone")

    save_output_image(composed.image, composed.dpi, output_image_file)
    save_output_image(composed.image_with_seams, composed.dpi, output_image_with_seams_file)
    save_output_image(composed.image_with_seam_areas, composed.dpi, output_image_with_seam_areas_file)
