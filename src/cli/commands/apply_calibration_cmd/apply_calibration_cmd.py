import argparse
from pathlib import Path
from typing import Protocol

from cli.commands.utils import save_output_image, load_scan_images
from scanner.calibration import apply_scanner_calibration, ScannerCalibration


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "apply-calibration",
        help="Applies the current calibration to the specified scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "scan_files",
        metavar="SCAN_FILES",
        nargs="+",
        help="The scan files or file patterns to apply the calibration to.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        metavar="OUTPUT_DIR",
        type=Path,
        default=Path(),
        help="Path to save the corrected images to.",
    )

    parser.set_defaults(handler=_run)


class ApplyCalibrationArgs(Protocol):
    scan_files: list[str]
    output_dir: Path


def _run(args: ApplyCalibrationArgs) -> None:
    scan_images = load_scan_images(args.scan_files)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scan_file in scan_images.scan_files:
        output_image_file = args.output_dir / f"corrected_{scan_file.name}"
        output_image_file.unlink(missing_ok=True)

    scanner_calibration = ScannerCalibration()
    corrected_scan_images = apply_scanner_calibration(scan_images, scanner_calibration)

    for scan_file, corrected_image, corrected_image_dpi in corrected_scan_images.files_and_images_and_dpi:
        output_image_file = args.output_dir / f"corrected_{scan_file.name}"
        save_output_image(corrected_image, corrected_image_dpi, output_image_file)
