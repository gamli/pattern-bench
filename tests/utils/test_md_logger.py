import importlib.util
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote

from src.utils.md_logger import MarkdownLogger, MdLogger


class ImageParser(HTMLParser):
    def __init__(self, content: str) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.feed(content)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


class _MdFixture(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nested" / "scan.md"
        self.logger = MdLogger(self.path)

    def content(self) -> str:
        return self.path.read_text(encoding="utf-8")


# noinspection none-function-assignment
class MdLoggerTests(_MdFixture):
    def test_creation_and_complete_blocks_without_context_or_close(self) -> None:
        self.assertTrue(self.path.is_file())
        self.assertIs(MarkdownLogger, MdLogger)
        self.assertIsNone(self.logger.h1("Scan"))
        self.assertIsNone(self.logger.p(content="**12 Teile** erkannt."))
        self.assertIsNone(self.logger.h2(content="Ergebnis"))
        self.assertEqual(
            self.content(), "# Scan\n\n**12 Teile** erkannt.\n\n## Ergebnis\n\n"
        )
        # On Windows this also detects a persistent open file handle.
        moved = self.path.with_name("renamed.md")
        self.path.rename(moved)
        moved.rename(self.path)
        self.logger.p("weiter")
        self.assertTrue(self.content().endswith("weiter\n\n"))

    def test_all_heading_levels(self) -> None:
        for level in range(1, 7):
            method = getattr(self.logger, f"h{level}")
            self.assertIsNone(method(f"Ebene {level}"))
            self.assertIn(f"{'#' * level} Ebene {level}\n\n", self.content())

    def test_reopen_preserves_existing_bytes_and_only_adds_one_title(self) -> None:
        original = "Bestehender Inhalt ä\r\n".encode()
        self.path.write_bytes(original)
        reopened = MdLogger(self.path, title="nicht erneut schreiben")
        reopened.p("fortgesetzt")
        self.assertTrue(self.path.read_bytes().startswith(original))
        self.assertEqual(self.path.read_bytes(), original + b"\nfortgesetzt\n\n")
        self.assertNotIn("nicht erneut schreiben", self.content())
        other = self.path.with_name("titled.md")
        MdLogger(other, title="Diagnose").h2("Erstes Ergebnis")
        MdLogger(other, title="Doppelter Titel").p("Zweites Ergebnis")
        self.assertEqual(
            other.read_text(encoding="utf-8"),
            "# Diagnose\n\n## Erstes Ergebnis\n\nZweites Ergebnis\n\n",
        )

    def test_raw_is_verbatim_and_next_block_adds_a_separator(self) -> None:
        self.logger.raw("**inline**")
        self.logger.raw(" und *mehr*")
        self.assertEqual(self.content(), "**inline** und *mehr*")
        self.logger.hr()
        self.logger.p("Ende")
        self.assertEqual(self.content(), "**inline** und *mehr*\n\n---\n\nEnde\n\n")

    def test_plain_text_escapes_markdown_html_and_block_markers(self) -> None:
        self.logger.text("# Titel\n- [Link](datei) **fett** <script>&\n    code\n===")
        self.assertEqual(
            self.content(),
            "\\# Titel  \n\\- \\[Link\\]\\(datei\\) \\*\\*fett\\*\\* "
            "&lt;script&gt;&amp;  \n"
            "&#32;&#32;&#32;&#32;code  \n\\=\\=\\=\n\n",
        )
        self.logger.text(42)
        self.assertTrue(self.content().endswith("42\n\n"))

    def test_code_fences_cannot_be_closed_by_their_content(self) -> None:
        code = "print('ä')\n```\n````\ntrailing spaces  "
        self.assertIsNone(self.logger.code(code, language="python"))
        self.assertEqual(self.content(), f"`````python\n{code}\n`````\n\n")
        self.logger.pre("")
        self.assertTrue(self.content().endswith("```\n```\n\n"))
        self.logger.code("with newline\n")
        self.assertTrue(self.content().endswith("```\nwith newline\n```\n\n"))

    def test_complete_lists_support_generators_and_multiline_items(self) -> None:
        self.assertIsNone(
            self.logger.ul(item for item in ["**erstens**", "zweitens\nweiter"])
        )
        self.assertIsNone(self.logger.ol(["acht", "neun\nweiter"], start=8))
        self.assertEqual(
            self.content(),
            "- **erstens**\n- zweitens\n  weiter\n\n8. acht\n9. neun\n   weiter\n\n",
        )
        before = self.content()
        self.logger.ul([])
        self.logger.ol([])
        self.assertEqual(self.content(), before)

    def test_table_cells_escape_markup_pipes_and_newlines(self) -> None:
        self.assertIsNone(
            self.logger.table(
                ["Name", "Wert"],
                (["A|B", "<b>Text</b>\n**raw**"], ["Missing", None]),
            )
        )
        self.assertEqual(
            self.content(),
            "| Name | Wert |\n| --- | --- |\n"
            "| A\\|B | &lt;b&gt;Text&lt;/b&gt;<br>\\*\\*raw\\*\\* |\n"
            "| Missing |  |\n\n",
        )

    def test_quote_and_html_elements_always_finish_in_one_call(self) -> None:
        self.logger.quote("erste Zeile\n\n**letzte**")
        self.assertIsNone(self.logger.tag("div", content="<b>raw</b>", class_="box"))
        self.assertIsNone(self.logger.div())
        self.assertIsNone(self.logger.span(content="ok", title='"<&>'))
        self.logger.tag(
            "input", type="checkbox", name="field", checked=True, disabled=False
        )
        self.logger.br()
        self.assertIn(
            '> erste Zeile\n> \n> **letzte**\n\n<div class="box"><b>raw</b></div>\n\n',
            self.content(),
        )
        self.assertIn("<div></div>\n\n", self.content())
        self.assertIn('<span title="&quot;&lt;&amp;&gt;">ok</span>', self.content())
        self.assertIn('<input type="checkbox" name="field" checked>', self.content())
        self.assertTrue(self.content().endswith("<br>\n\n"))

    def test_validation_errors_do_not_append_partial_blocks(self) -> None:
        self.logger.p("existing")
        before = self.path.read_bytes()
        operations = (
            lambda: self.logger.heading("title", level=0),
            lambda: self.logger.heading("title", level=7),
            lambda: self.logger.heading("title", level=True),
            lambda: self.logger.h1("line one\nline two"),
            lambda: self.logger.code("code", language="python\n```"),
            lambda: self.logger.ul("one string"),
            lambda: self.logger.ol(["value"], start=-1),
            lambda: self.logger.ol(["value"], start=True),
            lambda: self.logger.table([], []),
            lambda: self.logger.table(["A"], [[1], [2, 3]]),
            lambda: self.logger.tag('div onclick="bad"', content="bad"),
            lambda: self.logger.tag("img", content="not empty"),
            lambda: self.logger.div(content="no", **{"class": "a", "class_": "b"}),
            lambda: self.logger.div(content="no", **{'bad"attr': "bad"}),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises((ValueError, TypeError)):
                    operation()
                self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_title_does_not_create_a_file(self) -> None:
        path = self.path.with_name("invalid.md")
        with self.assertRaises(ValueError):
            MdLogger(path, title="one\ntwo")
        self.assertFalse(path.exists())

    def test_writes_from_one_instance_do_not_interleave_blocks(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(
                executor.map(lambda number: self.logger.p(f"block-{number}"), range(40))
            )
        self.assertEqual(
            set(self.content().strip().split("\n\n")),
            {f"block-{number}" for number in range(40)},
        )


class MdLoggerImageSavingTests(_MdFixture):
    """Test MdLogger's image-file handling without OpenCV or NumPy."""

    def test_saved_path_and_existing_files_are_preserved(self) -> None:
        original = self.path.parent / "scan_image_000001.png"
        original.write_bytes(b"existing image")
        path = self.logger._save_image(b"new image", ".png")
        self.assertIsInstance(path, Path)
        self.assertEqual(path, self.path.parent / "scan_image_000002.png")
        self.assertEqual(path.read_bytes(), b"new image")
        self.assertEqual(original.read_bytes(), b"existing image")

    def test_partial_files_are_removed_and_write_errors_propagate(self) -> None:
        original_open = Path.open
        for error_type in (OSError, FileExistsError):
            with self.subTest(error_type=error_type):
                failure = error_type("write failed after file creation")

                @contextmanager
                def failing_file(path):
                    with original_open(path, "xb") as stream:
                        def fail(data):
                            stream.write(data[:3])
                            raise failure

                        yield SimpleNamespace(write=fail)

                def open_file(path, mode="r", *args, **kwargs):
                    if mode == "xb":
                        return failing_file(path)
                    return original_open(path, mode, *args, **kwargs)

                with patch.object(Path, "open", new=open_file):
                    with self.assertRaises(error_type) as raised:
                        self.logger._save_image(b"partially written image", ".png")
                self.assertIs(raised.exception, failure)
                self.assertEqual(list(self.path.parent.glob("*.png")), [])

    def test_open_errors_do_not_remove_existing_images(self) -> None:
        original_open = Path.open
        existing = self.path.parent / "scan_image_000001.png"
        existing.write_bytes(b"keep this image")

        def open_file(path, mode="r", *args, **kwargs):
            if mode == "xb" and path != existing:
                raise PermissionError("cannot create next image")
            return original_open(path, mode, *args, **kwargs)

        with patch.object(Path, "open", new=open_file):
            with self.assertRaises(PermissionError):
                self.logger._save_image(b"new image", ".png")
        self.assertEqual(existing.read_bytes(), b"keep this image")
        self.assertEqual(list(self.path.parent.glob("*.png")), [existing])


@unittest.skipUnless(
    importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
    "Image tests require OpenCV and NumPy",
)
class MdImageTests(_MdFixture):
    def setUp(self) -> None:
        super().setUp()
        import numpy as np

        self.np = np
        self.image = np.full((8, 12, 3), (10, 80, 240), dtype=np.uint8)

    def assert_image_roundtrip(self, path: Path, image: object) -> None:
        import cv2

        decoded = cv2.imdecode(
            self.np.frombuffer(path.read_bytes(), dtype=self.np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        self.np.testing.assert_array_equal(decoded, image)

    def test_image_exports_without_context_and_preserves_html_features(self) -> None:
        import cv2

        self.assertIsNone(
            self.logger.img(
                self.image,
                alt='Original <&> "ä"',
                width=640,
                class_="preview",
                params=[cv2.IMWRITE_PNG_COMPRESSION, 9],
                link_attrs={"class_": "full-size"},
            )
        )
        parser = ImageParser(self.content())
        self.assertEqual([tag for tag, _ in parser.elements], ["a", "img"])
        link, img = [attrs for _, attrs in parser.elements]
        self.assertEqual(link["target"], "_blank")
        self.assertEqual(link["class"], "full-size")
        self.assertEqual(link["href"], img["src"])
        self.assertEqual(img["alt"], 'Original <&> "ä"')
        self.assertEqual(img["width"], "640")
        self.assertEqual(img["class"], "preview")
        self.assert_image_roundtrip(
            self.path.parent / unquote(img["src"] or ""), self.image
        )

    def test_pure_markdown_image_is_linked_and_escapes_alt_text(self) -> None:
        self.assertIsNone(
            self.logger.img(self.image, markdown=True, alt="Original [1] *ä*")
        )
        self.assertEqual(
            self.content(),
            "[![Original \\[1\\] \\*ä\\*](scan_image_000001.png)]"
            "(scan_image_000001.png)\n\n",
        )
        self.assertNotIn("<", self.content())
        self.assert_image_roundtrip(
            self.path.parent / "scan_image_000001.png", self.image
        )

    def test_image_numbering_resumes_and_skips_collisions(self) -> None:
        first = self.path.parent / "scan_image_000001.png"
        first.write_bytes(b"keep newly created file")
        self.logger.img(self.image)
        seventh = self.path.parent / "scan_image_000007.jpg"
        seventh.write_bytes(b"keep old file")
        MdLogger(self.path).img(self.image, extension="PNG", markdown=True)
        self.assertEqual(first.read_bytes(), b"keep newly created file")
        self.assertEqual(seventh.read_bytes(), b"keep old file")
        self.assert_image_roundtrip(
            self.path.parent / "scan_image_000002.png", self.image
        )
        self.assert_image_roundtrip(
            self.path.parent / "scan_image_000008.png", self.image
        )
        self.assertIn("scan_image_000002.png", self.content())
        self.assertIn("scan_image_000008.png", self.content())

    def test_special_paths_and_image_shapes(self) -> None:
        path = self.path.parent / "Umläute & [test]" / "Größe #1.md"
        logger = MdLogger(path)
        images = (
            self.np.arange(24, dtype=self.np.uint16).reshape(4, 6),
            self.np.full((4, 6, 4), (20, 80, 140, 100), dtype=self.np.uint8),
        )
        for image in images:
            logger.img(image, markdown=True)
        urls = re.findall(r"!\[]\(([^)]+)\)", path.read_text(encoding="utf-8"))
        self.assertEqual(len(urls), 2)
        for image, url in zip(images, urls):
            self.assertNotIn("#", url)
            self.assertNotIn(" ", url)
            self.assert_image_roundtrip(path.parent / unquote(url), image)

    def test_invalid_images_or_attributes_leave_no_files_or_entries(self) -> None:
        before_files = set(self.path.parent.iterdir())
        cases = (
            (self.np.zeros((0, 2), dtype=self.np.uint8), {}),
            (self.np.zeros((2, 2, 2), dtype=self.np.uint8), {}),
            (self.image, {"extension": "../png"}),
            (self.image, {"extension": ".noencoder"}),
            (self.image, {"src": "other.png"}),
            (self.image, {"link_attrs": {"target": "_self"}}),
            (self.image, {"markdown": True, "width": 100}),
            (self.image, {"markdown": True, "link_attrs": {"class_": "x"}}),
        )
        for image, attrs in cases:
            with self.assertRaises(ValueError):
                self.logger.img(image, **attrs)
        with self.assertRaises(TypeError):
            self.logger.img([[1, 2]])
        with patch("cv2.imencode", return_value=(False, self.np.array([], dtype=self.np.uint8))):
            with self.assertRaises(ValueError):
                self.logger.img(self.image)
        self.assertEqual(self.content(), "")
        self.assertEqual(set(self.path.parent.iterdir()), before_files)


if __name__ == "__main__":
    unittest.main()
