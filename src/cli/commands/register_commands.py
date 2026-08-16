import argparse

from cli.commands.apply_calibration_cmd import apply_calibration_cmd
from src.cli.commands import stitch_scans_cmd


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    stitch_scans_cmd.register(subparsers)
    apply_calibration_cmd.register(subparsers)
