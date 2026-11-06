import argparse
import shutil
from pathlib import Path
from typing import Protocol

from scanner.calibration import apply_scanner_calibration, ScannerCalibration
from stitching.shared import StitchingConfig
from stitching.composition import compose_images
from stitching.transformation_estimation import estimate_global_transforms
from stitching.feature_detection import detect_features
from stitching.feature_matching import match_features
from utils.md_logger import MdLogger

from ..utils import save_output_image, load_scan_images


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

    parser.set_defaults(handler=_run)


class StitchScansArgs(Protocol):
    scan_files: list[str]
    output_dir: Path
    verbose: bool
    use_fixed_seed: bool


def _run(args: StitchScansArgs) -> None:
    stitching_images = load_scan_images(args.scan_files)

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
    verbose_dir.mkdir()

    vlog = MdLogger(verbose_dir / "log.md") if args.verbose else None

    config = StitchingConfig(
        vlog=vlog,
        use_fixed_seed=args.use_fixed_seed,
    )

    scanner_calibration = ScannerCalibration()
    corrected_stitching_images = apply_scanner_calibration(stitching_images, scanner_calibration, config.vlog)
    features = detect_features(corrected_stitching_images, config)
    pairwise_matches = match_features(features, corrected_stitching_images, config)
    image_transforms = estimate_global_transforms(pairwise_matches, corrected_stitching_images, config)
    composed = compose_images(image_transforms, corrected_stitching_images, config)

    save_output_image(composed.image, composed.dpi, output_image_file)
    save_output_image(composed.image_with_seams, composed.dpi, output_image_with_seams_file)
    save_output_image(composed.image_with_seam_areas, composed.dpi, output_image_with_seam_areas_file)
