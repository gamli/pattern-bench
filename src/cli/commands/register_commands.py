import argparse

from . import apply_calibration_cmd, stitch_scans_cmd


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    stitch_scans_cmd.register(subparsers)
    apply_calibration_cmd.register(subparsers)
