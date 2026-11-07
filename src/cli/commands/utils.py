import glob
import sys
import time
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Sequence, cast, Any

import cv2
import numpy as np
import psutil
from PIL import Image

from stitching.shared import ImageArray, StitchingImages

def start_info_timer():
    return InfoTimer()

class InfoTimer:
    def __init__(self):
        self._start = time.perf_counter()
        self._last_print = self._start
    def print(self, prefix_txt: str):
        current = time.perf_counter()
        total = current - self._start
        since_last_print = current - self._last_print
        print_info(f"{prefix_txt}: {since_last_print:.2f} s (total: {total:.2f} s)")
        self._last_print = current

def print_peak_memory(prefix_txt: str) -> None:
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", -1)
    print_info(f"[[[ {prefix_txt}: Max RAM {peak / 1024**3:.2f} GiB ]]]")

def print_info(txt: str) -> None:
    print(txt, file=sys.stderr)

def save_output_image(image: ImageArray, dpi: float | None, file: Path) -> None:
    image_array = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = Image.fromarray(image_array)
    if dpi:
        image.save(file, dpi=(dpi, dpi))
    else:
        image.save(file)


def resolve_scan_files(scan_files_patterns: Sequence[str]) -> list[Path]:
    resolved_files: list[Path] = []
    for pattern in scan_files_patterns:
        matched = [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]
        if matched:
            for p in matched:
                if p.is_file() and p not in resolved_files:
                    resolved_files.append(p)
    return resolved_files


def load_scan_images(scan_files: list[str]) -> StitchingImages:
    scan_files = resolve_scan_files(scan_files)
    if not scan_files:
        raise ValueError("No scan files found.")

    images = []
    for file_path in scan_files:
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        images.append(image)

    stitching_images = StitchingImages(
        scan_files=tuple(scan_files),
        scan_images=tuple(images),
        scan_dpis=tuple(repeat(None, len(scan_files))),
    )

    return stitching_images
