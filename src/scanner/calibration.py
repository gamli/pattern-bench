import math
from configparser import ConfigParser
from numbers import Real
from pathlib import Path
from typing import Literal, Sequence, cast

import cv2 as cv
import numpy as np
from PIL import Image

import utils.global_paths as gp
from scanner.constants import SCANNER_DPI, AVAILABLE_SCANNER_DPIS
from stitching.shared import StitchingImages, ImageArray
from utils.md_logger import MdLogger

_SCALE_X_VAR = "scale_x"
_SCALE_Y_VAR = "scale_y"


class ScannerCalibration:

    def __init__(self) -> None:
        self._by_dpi: dict[SCANNER_DPI, tuple[float, float]] = {dpi: (1., 1.) for dpi in AVAILABLE_SCANNER_DPIS}
        self.load()

    def scale(self, dpi: SCANNER_DPI) -> tuple[float, float]:
        return self.scale_x(dpi), self.scale_y(dpi)

    def set_scale(self, dpi: SCANNER_DPI, scale: tuple[float, float]) -> None:
        scale_x, scale_y = scale
        self._validate_scale_value(dpi, "x", scale_x)
        self._validate_scale_value(dpi, "y", scale_y)
        self._by_dpi[dpi] = scale_x, scale_y

    def scale_x(self, dpi: SCANNER_DPI) -> float:
        return self._by_dpi[dpi][0]

    def set_scale_x(self, dpi: SCANNER_DPI, scale_x: float) -> None:
        self._validate_scale_value(dpi, "x", scale_x)
        _, scale_y = self._by_dpi[dpi]
        self._by_dpi[dpi] = scale_x, scale_y

    def scale_y(self, dpi: SCANNER_DPI) -> float:
        return self._by_dpi[dpi][1]

    def set_scale_y(self, dpi: SCANNER_DPI, scale_y: float) -> None:
        self._validate_scale_value(dpi, "y", scale_y)
        scale_x, _ = self._by_dpi[dpi]
        self._by_dpi[dpi] = scale_x, scale_y

    def load(self) -> None:

        self._assert_calibration_file_exists()

        by_dpi: dict[SCANNER_DPI, tuple[float, float]] = {}

        config = ConfigParser()
        with gp.calibration_file.open(encoding="utf-8-sig") as file:
            config.read_file(file)

        for dpi in AVAILABLE_SCANNER_DPIS:
            section = self._dpi_section(dpi)
            scale_x = config.getfloat(section, _SCALE_X_VAR, fallback=1.)
            self._validate_scale_value(dpi, "x", scale_x)
            scale_y = config.getfloat(section, _SCALE_Y_VAR, fallback=1.)
            self._validate_scale_value(dpi, "y", scale_y)
            by_dpi[dpi] = (scale_x, scale_y)

        self._by_dpi = by_dpi

    def save(self) -> None:

        self._assert_calibration_file_exists()

        config = ConfigParser()
        with gp.calibration_file.open(encoding="utf-8-sig") as file:
            config.read_file(file)

        for dpi in AVAILABLE_SCANNER_DPIS:
            section = self._dpi_section(dpi)
            scale_x, scale_y = self._by_dpi[dpi]
            config[section] = {
                _SCALE_X_VAR: str(scale_x),
                _SCALE_Y_VAR: str(scale_y),
            }

        with gp.calibration_file.open("w", encoding="utf-8-sig") as file:
            config.write(file)

    @staticmethod
    def _validate_scale_value(dpi: SCANNER_DPI, x_or_y: Literal["x", "y"], scale_value: float) -> None:
        if not (math.isfinite(scale_value) and scale_value > 0):
            raise ValueError(f"Invalid scale_{x_or_y} value for {dpi} DPI: {scale_value}")

    @staticmethod
    def _dpi_section(dpi: SCANNER_DPI) -> str:
        return f"{dpi}dpi"

    @staticmethod
    def _assert_calibration_file_exists() -> None:
        if not gp.calibration_file.exists():
            gp.calibration_file.parent.mkdir(parents=True, exist_ok=True)
            Path(__file__).with_name("default-calibration.ini").copy(gp.calibration_file)


def apply_scanner_calibration(
        stitching_images: StitchingImages,
        scanner_calibration: ScannerCalibration,
        vlog: MdLogger | None = None,
) -> StitchingImages:
    """Apply ScannerCalibration.scale(dpi) before feature detection.

    The supplied factors correct the scanner's sensor axis (x) and travel
    direction (y), in its original raster orientation. They are measured
    elsewhere; this step only applies them. Subsequent stages use the returned
    images and their coordinates. Input arrays are never modified.
    """

    if vlog:
        vlog.h3("Apply Scanner Calibration")
        vlog.start_timer("Apply Scanner Calibration (total)")
        vlog.log_elapsed("Applying scanner calibration starting at")

    try:
        corrected_images: list[ImageArray] = []
        image_dpis: list[SCANNER_DPI | None] = []
        for file, image in stitching_images.files_and_images:

            if vlog:
                vlog.p(f"Applying calibration to image `{file.name}`")
                vlog.start_timer("Applying calibration")

            image_dpi = _read_dpi(file)

            if image_dpi is None:
                if vlog:
                    vlog.p("Image has unsupported DPI; using the input image directly.")
                corrected_images.append(image)
                image_dpis.append(None)
                continue

            image_calibration_scale = scanner_calibration.scale(image_dpi)

            if image_calibration_scale == (1., 1.):
                if vlog:
                    vlog.p("Both factors are 1; using the input image directly.")
                corrected_images.append(image)
                image_dpis.append(image_dpi)
                continue

            if vlog:
                vlog.p(
                    f"Applying Scale-X=`{image_calibration_scale[0]:.9f}`, Scale-Y=`{image_calibration_scale[1]:.9f}`. "
                    "x follows the sensor axis; y follows the scan direction.")

            height, width = image.shape[:2]
            # Only the canvas is rounded. Sampling uses the exact factors.
            # nextafter avoids an extra pixel for e.g. 50 * 1.1 > 55 by one ULP.
            size = (max(1, math.ceil(np.nextafter(width * image_calibration_scale[0], -np.inf))),
                    max(1, math.ceil(np.nextafter(height * image_calibration_scale[1], -np.inf))))
            corrected_image = cv.warpAffine(
                image,
                np.diag((*image_calibration_scale, 1.))[:2],
                size,
                flags=cv.INTER_LINEAR,
                borderMode=cv.BORDER_REPLICATE,
            )

            corrected_images.append(corrected_image)
            image_dpis.append(image_dpi)

            if vlog:
                vlog.p(f"`{file.name}`: `{width} x {height}` -> `{size[0]} x {size[1]}` px.")
                vlog.end_timer()
                vlog.img(corrected_image, alt="Corrected Image", style="max-height: 500px;")

        return StitchingImages(
            stitching_images.scan_files,
            tuple(corrected_images),
            tuple(image_dpis),
        )

    finally:
        if vlog:
            vlog.end_timer()
            vlog.log_elapsed("Applying scanner calibration finished at")


def _read_dpi(file: str | Path) -> SCANNER_DPI | None:
    with Image.open(file) as image:
        raw_dpi: object = image.info.get("dpi")

    values: Sequence[object] = raw_dpi if isinstance(raw_dpi, Sequence) else (raw_dpi,)

    if len(values) not in (1, 2):
        return None

    first = values[0]

    if not isinstance(first, Real):
        return None

    if len(values) == 2:
        second: object = values[1]
        if not isinstance(second, Real) or first != second:
            return None

    dpi = float(first)

    return cast(SCANNER_DPI, int(dpi)) if dpi in AVAILABLE_SCANNER_DPIS else None
