"""Small logging helpers shared by the stitching stages."""

from utils.md_logger import MdLogger


def _vlog_start_timer(vlog: MdLogger | None, label: str, *label_args: object) -> None:
    if vlog:
        vlog.start_timer(label.format(*label_args) if label_args else label)


def _vlog_end_timer(vlog: MdLogger | None) -> None:
    if vlog:
        vlog.end_timer()


def _vlog_error(vlog: MdLogger | None, message: object) -> None:
    if vlog:
        # noinspection string-conversion-without-dunder-method
        vlog.err(str(message))
