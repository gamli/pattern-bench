"""Append-only Markdown logs with local OpenCV images; no with/close needed.

Every operation opens and closes the file. Only img() needs OpenCV and NumPy.
One logger instance should write to a file at a time; its methods are serialized.
"""

import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from pathlib import Path
from textwrap import dedent
from threading import RLock, local
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["MdLogger", "MarkdownLogger"]

AttributeValue = str | int | float | bool | None
_TAG_NAME = re.compile(r"[A-Za-z][A-Za-z0-9:_-]*\Z")
_ATTRIBUTE_NAME = re.compile(r"[A-Za-z_:][A-Za-z0-9_:.-]*\Z")
_MARKDOWN_PUNCTUATION = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~=])")
_VOID_TAGS = frozenset(
    "area base br col embed hr img input link meta param source track wbr".split()
)


def _plain(value: object) -> str:
    """Escape Markdown syntax and HTML, including indented block starts."""
    # noinspection string-conversion-without-dunder-method
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = _MARKDOWN_PUNCTUATION.sub(r"\\\1", escape(text, quote=False))

    def escape_indent(match: re.Match[str]) -> str:
        return match.group(0).replace(" ", "&#32;").replace("\t", "&#9;")

    return re.sub(r"(?m)^[ \t]+", escape_indent, text)


def _attribute_name(name: str) -> str:
    name = name.removesuffix("_").replace("_", "-")
    if not _ATTRIBUTE_NAME.fullmatch(name):
        raise ValueError(f"Invalid HTML attribute name: {name!r}")
    return name


def _attributes(values: Mapping[str, AttributeValue]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for key, value in values.items():
        name = _attribute_name(key)
        if name.lower() in seen:
            raise ValueError(f"Duplicate HTML attribute: {name!r}")
        seen.add(name.lower())
        if value is None or value is False:
            continue
        if value is True:
            parts.append(f" {name}")
        else:
            parts.append(f' {name}="{escape(str(value), quote=True)}"')
    return "".join(parts)


class MdLogger:
    """Create or continue a UTF-8 Markdown log without keeping a file open.

    p(), headings and list items accept Markdown. text() escapes plain text;
    raw() appends verbatim. Other operations write complete, separated blocks
    and return None, except timing methods that return elapsed seconds.
    HTML tag()/div()/span() also always write complete elements.
    """

    def __init__(
            self,
            path_to_log_file: str | os.PathLike[str],
            *,
            title: str | None = None,
    ) -> None:
        if title is not None:
            self._check_heading(title, 1)
        self._started_at = time.perf_counter()
        self._timer_state = local()
        self.path = Path(path_to_log_file).expanduser().resolve()
        self._lock = RLock()
        self._image_prefix = f"{self.path.stem}_image_"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as stream:
            if stream.tell() == 0:
                stream.write(dedent(
                    """
                    <style>
                    .img-link {
                        display: inline-flex;
                        flex-direction: column;
                        align-items: start;
                        border: 2px solid lime;
                        padding: 2px;
                        margin: 2px;
                    }
                    .img-link:hover {
                        cursor: pointer;
                        filter: drop-shadow(0 0 8px green);
                    }
                    .img-link > img {
                        max-width: 100%;
                        height: auto;
                    }
                    </style>
                    """).lstrip().encode())
                if title is not None:
                    stream.write(f"# {title}\n\n".encode())

        pattern = re.compile(
            re.escape(self._image_prefix) + r"([0-9]+)\.[A-Za-z0-9]+\Z"
        )
        self._next_image_number = 1 + max(
            (
                int(match[1])
                for entry in self.path.parent.iterdir()
                if (match := pattern.fullmatch(entry.name))
            ),
            default=0,
        )

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed real seconds since construction, using a monotonic clock."""
        return time.perf_counter() - self._started_at

    def log_elapsed(self, label: str = "Total elapsed time") -> float:
        """Write the elapsed time since construction and return seconds."""
        elapsed = self.elapsed_seconds
        self.p(f"{label}: `{elapsed:.3f} s`")
        return elapsed

    def start_timer(self, label: str) -> None:
        """Start a named timer without logging; calls nest within each thread."""
        stack: list[tuple[str, float | int]] | None = getattr(self._timer_state, "stack", None)
        if stack is None:
            stack = []
            self._timer_state.stack = stack
        # noinspection unresolved-references
        stack.append((label, time.perf_counter()))

    def end_timer(self) -> float:
        """Log and return seconds for this thread's innermost timer.

        The duration includes nested measurements and excludes writing this
        timer's log entry. A failed write leaves the timer active for retry.
        """
        ended_at = time.perf_counter()
        stack: list[tuple[str, float | int]] | None = getattr(self._timer_state, "stack", None)
        if not stack:
            raise RuntimeError("end_timer() requires a matching start_timer() in this thread.")
        label, started_at = stack[-1]
        elapsed = ended_at - started_at
        self.p(f"{label}: `{elapsed:.3f} s`")
        stack.pop()
        return elapsed

    def _append(self, content: str, *, block: bool = True) -> None:
        # Encode first so invalid text cannot leave a partially appended block.
        data = (content.rstrip("\r\n") + "\n\n" if block else content).encode("utf-8")
        with self._lock, self.path.open("a+b") as stream:
            prefix = b""
            size = stream.seek(0, os.SEEK_END)
            if block and size:
                stream.seek(max(0, size - 4))
                tail = stream.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                if not tail.endswith(b"\n\n"):
                    prefix = b"\n" if tail.endswith(b"\n") else b"\n\n"
            # Append mode always writes at EOF, including after the tail read.
            stream.write(prefix + data)

    def raw(self, markdown: str) -> None:
        """Append exactly this Markdown/HTML, with no automatic line breaks."""
        self._append(markdown, block=False)

    def p(self, content: str) -> None:
        """Write a paragraph/block of Markdown unchanged."""
        self._append(content)

    def text(self, value: object) -> None:
        """Write plain text, escaping Markdown and HTML; preserve line breaks."""
        self._append(_plain(value).replace("\n", "  \n"))

    @staticmethod
    def _check_heading(content: str, level: int) -> None:
        if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 6:
            raise ValueError("Heading level must be an integer from 1 to 6.")
        if not isinstance(content, str):
            raise TypeError("Heading content must be a string.")
        if "\n" in content or "\r" in content:
            raise ValueError("Heading content must fit on one line.")

    def heading(self, content: str, level: int = 1) -> None:
        """Write a Markdown heading. Its content may contain inline Markdown."""
        self._check_heading(content, level)
        self._append(f"{'#' * level} {content}")

    def h1(self, content: str) -> None:
        self.heading(content, 1)

    def h2(self, content: str) -> None:
        self.heading(content, 2)

    def h3(self, content: str) -> None:
        self.heading(content, 3)

    def h4(self, content: str) -> None:
        self.heading(content, 4)

    def h5(self, content: str) -> None:
        self.heading(content, 5)

    def h6(self, content: str) -> None:
        self.heading(content, 6)

    def hr(self) -> None:
        self._append("---")

    def quote(self, content: str) -> None:
        self._append("\n".join(f"> {line}" for line in content.split("\n")))

    def code(self, content: str, *, language: str = "") -> None:
        """Write a complete fenced code block, safely containing backticks."""
        if not re.fullmatch(r"[A-Za-z0-9_+#.\-]*", language):
            raise ValueError("language must be a simple name, e.g. 'python' or 'c++'.")
        longest_run = max(
            (len(match[0]) for match in re.finditer(r"`+", content)), default=0
        )
        fence = "`" * max(3, longest_run + 1)
        newline = "" if content.endswith("\n") or not content else "\n"
        self._append(f"{fence}{language}\n{content}{newline}{fence}")

    def pre(self, content: str) -> None:
        self.code(content)

    def _list(self, items: Iterable[object], *, start: int | None) -> None:
        if isinstance(items, (str, bytes)):
            raise TypeError("Pass an iterable of list items, not a single string.")
        lines: list[str] = []
        for index, item in enumerate(items):
            marker = "-" if start is None else f"{start + index}."
            if len(marker) > 10:
                raise ValueError("Ordered list numbers must have at most 9 digits.")
            parts = str(item).replace("\r\n", "\n").replace("\r", "\n").split("\n")
            lines.append(f"{marker} {parts[0]}")
            lines.extend(f"{' ' * (len(marker) + 1)}{line}" for line in parts[1:])
        if lines:
            self._append("\n".join(lines))

    def ul(self, items: Iterable[object]) -> None:
        """Write a complete unordered list. Items may contain Markdown."""
        self._list(items, start=None)

    def ol(self, items: Iterable[object], *, start: int = 1) -> None:
        """Write a complete ordered list, optionally starting at another number."""
        if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or not 0 <= start < 10 ** 9
        ):
            raise ValueError("start must be an integer from 0 to 999999999.")
        self._list(items, start=start)

    def table(
            self, headers: Sequence[object], rows: Iterable[Sequence[object]]
    ) -> None:
        """Write a GFM table, escaping cell text, pipes and embedded line breaks."""
        if isinstance(headers, (str, bytes)) or not headers:
            raise ValueError("Provide a nonempty sequence of table headers.")

        def cell(value: object) -> str:
            return _plain("" if value is None else value).replace("\n", "<br>")

        def row(values: Sequence[object]) -> str:
            if isinstance(values, (str, bytes)) or len(values) != len(headers):
                raise ValueError("Each table row must have the same width as headers.")
            return "| " + " | ".join(cell(value) for value in values) + " |"

        lines = [row(headers), "| " + " | ".join("---" for _ in headers) + " |"]
        lines.extend(row(values) for values in rows)
        self._append("\n".join(lines))

    def tag(
            self,
            name: str,
            /,
            *,
            content: str | None = None,
            **attrs: AttributeValue,
    ) -> None:
        """Write one complete HTML element as a block; never opens a context.

        Content is raw HTML, not Markdown. CSS and class_ are supported here.
        Omitted/None content means an empty element, and returns None as well.
        """
        if not _TAG_NAME.fullmatch(name):
            raise ValueError(f"Invalid HTML tag name: {name!r}")
        if content is not None and not isinstance(content, str):
            raise TypeError("content must be a string or None.")
        name = name.lower()
        opening = f"<{name}{_attributes(attrs)}>"
        if name in _VOID_TAGS:
            if content:
                raise ValueError(f"Void element {name!r} cannot have content.")
            self._append(opening)
        else:
            self._append(f"{opening}{content or ''}</{name}>")

    def div(self, content: str | None = None, **attrs: AttributeValue) -> None:
        self.tag("div", content=content, **attrs)

    def span(self, content: str | None = None, **attrs: AttributeValue) -> None:
        self.tag("span", content=content, **attrs)

    def br(self) -> None:
        self.tag("br")

    def img(
            self,
            opencv_image: NDArray[Any],
            /,
            *,
            extension: str = ".png",
            params: Sequence[int] | None = None,
            link_attrs: Mapping[str, AttributeValue] | None = None,
            markdown: bool = False,
            link_only: bool = False,
            **attrs: AttributeValue,
    ) -> None:
        """Save a numbered image beside the log and write a clickable reference.

        By default HTML preserves target=_blank and image attributes exactly
        as in HtmlLogger. markdown=True writes [![alt](file)](file) instead;
        only alt is supported in that mode (no styles, width or link_attrs).
        link_only=True writes a text link instead of an image preview, using
        alt or the generated filename as its label. Image attributes apply
        only to previews; link_attrs also applies to HTML text links.
        Grayscale, BGR and BGRA arrays retain their OpenCV channel order.
        """
        image_attributes = _attributes(attrs)
        anchor_values = link_attrs if link_attrs is not None else {}
        anchor_attributes = _attributes(anchor_values)

        if any(_attribute_name(key).lower() == "src" for key in attrs):
            raise ValueError("img() manages the src attribute.")
        if any(_attribute_name(key).lower() in {"href", "target", "rel"} for key in anchor_values):
            raise ValueError("img() manages the link's href, target and rel.")
        if markdown and (anchor_values or any(_attribute_name(key).lower() != "alt" for key in attrs)):
            raise ValueError("markdown=True supports only alt; HTML attributes need HTML.")
        extension = "." + extension.lower().removeprefix(".")
        if not re.fullmatch(r"\.[a-z0-9]+", extension):
            raise ValueError(f"Invalid image extension: {extension!r}")

        if not isinstance(opencv_image, np.ndarray):
            raise TypeError("opencv_image must be a NumPy array.")
        if opencv_image.size == 0 or opencv_image.ndim not in (2, 3):
            raise ValueError("Expected a nonempty grayscale, BGR or BGRA image.")
        if opencv_image.ndim == 3 and opencv_image.shape[2] not in (1, 3, 4):
            raise ValueError("Expected an image with 1, 3 or 4 channels.")

        try:
            success, encoded = cv2.imencode(extension, opencv_image, list(params) if params is not None else [])
        except cv2.error as error:
            raise ValueError(f"Cannot encode image as {extension}: {error}") from error

        if not success:
            raise ValueError(f"Cannot encode image as {extension}.")

        with self._lock:
            image_path = self._save_image(encoded.tobytes(), extension)
            url = quote(image_path.name, safe="")
            alt = next(
                (value for key, value in attrs.items() if _attribute_name(key).lower() == "alt"),
                None,
            )
            label = str(alt) if alt is not None else ""
            if link_only:
                label = label or image_path.name
            label = label.replace("\r", " ").replace("\n", " ")
            if markdown:
                label = _plain(label)
                self._append(f"[{label}]({url})" if link_only else f"[![{label}]({url})]({url})")
            else:
                label_span = f'<span>{escape(label, quote=False)}</span>'
                content = label_span if link_only else f'{label_span}\n<img src="{url}"{image_attributes}>'
                self._append(
                    f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="img-link"'
                    f'{anchor_attributes}>\n{content}\n</a>'
                )

    def _save_image(self, data: bytes, extension: str) -> Path:
        number = self._next_image_number
        path = self.path.with_name(f"{self._image_prefix}{number:06d}{extension}")
        while True:
            created = False
            try:
                with path.open("xb") as stream:
                    created = True
                    stream.write(data)
            except BaseException as error:
                if created:
                    path.unlink(missing_ok=True)
                    raise
                if not isinstance(error, FileExistsError):
                    raise
                number += 1
                path = self.path.with_name(
                    f"{self._image_prefix}{number:06d}{extension}"
                )
            else:
                break

        self._next_image_number = number + 1
        return path

    def err(self, content: str | None = None) -> None:
        self.span(
            f'<span style="font-weight: bold; text-decoration: underline;">Error: </span>{content}',
            style="color: red",
        )


MarkdownLogger = MdLogger
