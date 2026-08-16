import argparse

from src.cli.commands import register_commands


def main() -> None:
    args = _build_argument_parser().parse_args()
    args.handler(args)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    command_parsers = parser.add_subparsers(dest="command", required=True)
    register_commands(command_parsers)
    return parser
