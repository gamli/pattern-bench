import argparse
import sys

from .commands import register_commands


def main() -> None:
    argv = sys.argv[1:] or ["stitch-scans"]
    args = _build_argument_parser().parse_args(argv)
    args.handler(args)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    command_parsers = parser.add_subparsers(dest="command", required=True)
    register_commands(command_parsers)
    return parser
