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
        nargs="*",
        help="The scan files or file patterns to stitch. Opens a file dialog if omitted.",
    )
    parser.add_argument(
        "-o", "--output-file",
        metavar="OUTPUT_FILE",
        type=Path,
        default=None,
        help="Path to save the stitched scan image to. Opens a save dialog if omitted.",
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
    scan_files: list[str] | None
    output_file: Path | None
    verbose: bool
    use_fixed_seed: bool
    no_pixel_refinement: bool


def _run(args: StitchScansArgs) -> None:
    scan_files_and_output_file = _scan_files_and_output_file(args)
    if scan_files_and_output_file is None:
        print_info("Cancelled.")
        return
    scan_files, output_file = scan_files_and_output_file

    print_info("Running stitch-scans")

    info_timer = start_info_timer()

    print_info("Loading images")
    stitching_images = load_scan_images(scan_files)
    info_timer.print("\tDone")

    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file.unlink(missing_ok=True)

    output_image_with_seams_file = output_file.with_stem(f"{output_file.stem}_with_seams")
    output_image_with_seams_file.unlink(missing_ok=True)

    output_image_with_seam_areas_file = output_file.with_stem(f"{output_file.stem}_with_seam_areas")
    output_image_with_seam_areas_file.unlink(missing_ok=True)

    verbose_dir = output_dir / "verbose"
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

    save_output_image(composed.image, composed.dpi, output_file)
    save_output_image(composed.image_with_seams, composed.dpi, output_image_with_seams_file)
    save_output_image(composed.image_with_seam_areas, composed.dpi, output_image_with_seam_areas_file)

def _scan_files_and_output_file(args: StitchScansArgs) -> tuple[list[str], Path] | None:

    scan_files = args.scan_files
    output_file = args.output_file

    scan_files = [file for file in (scan_files or []) if file.strip()]
    if scan_files and output_file is not None:
        return scan_files, output_file

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        if not scan_files:
            scan_files = list(filedialog.askopenfilenames(
                parent=root,
                title="Select scans to stitch",
                filetypes=scan_filetypes,
            ))
            if not scan_files:
                return None

        if output_file is None:
            selected_file = filedialog.asksaveasfilename(
                parent=root,
                title="Save stitched scan",
                initialfile="stitched_scans.jpg",
                defaultextension=".jpg",
                filetypes=output_filetypes,
            )
            if not selected_file:
                return None
            output_file = Path(selected_file)

        return scan_files, output_file
    finally:
        root.destroy()


scan_filetypes: list[tuple[str, str]] = [
    ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
    ("All files", "*"),
]

output_filetypes: list[tuple[str, str]] = [
    ("JPEG image", "*.jpg *.jpeg"),
    ("PNG image", "*.png"),
    ("TIFF image", "*.tif *.tiff"),
    ("BMP image", "*.bmp"),
    ("All files", "*"),
]
