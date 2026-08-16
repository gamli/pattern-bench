# import importlib.util
# import tempfile
# import unittest
# from contextlib import contextmanager
# from html.parser import HTMLParser
# from pathlib import Path
# from types import SimpleNamespace
# from unittest.mock import patch
# from urllib.parse import unquote
#
# from src.utils.html_logger import HtmlLogger
#
#
# class DocumentParser(HTMLParser):
#     """Check structural nesting and collect image/link attributes."""
#
#     def __init__(self) -> None:
#         super().__init__(convert_charrefs=True)
#         self.stack: list[str] = []
#         self.elements: list[tuple[str, dict[str, str | None]]] = []
#         self.errors: list[tuple[str, list[str]]] = []
#
#     def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
#         self.elements.append((tag, dict(attrs)))
#         if tag not in {"meta", "img", "input", "br", "hr"}:
#             self.stack.append(tag)
#
#     def handle_endtag(self, tag: str) -> None:
#         if not self.stack or self.stack[-1] != tag:
#             self.errors.append((tag, self.stack.copy()))
#         else:
#             self.stack.pop()
#
#
# class _LoggerFixture(unittest.TestCase):
#     def setUp(self) -> None:
#         self.directory = tempfile.TemporaryDirectory()
#         self.addCleanup(self.directory.cleanup)
#         self.path = Path(self.directory.name) / "nested" / "log.html"
#         self.logger = HtmlLogger(self.path, title='Log <&> "ä"')
#         self.addCleanup(self.logger.close)
#
#     def html(self) -> str:
#         return self.path.read_text(encoding="utf-8")
#
#     def parsed(self) -> DocumentParser:
#         parser = DocumentParser()
#         parser.feed(self.html())
#         self.assertEqual(parser.errors, [])
#         self.assertEqual(parser.stack, [])
#         return parser
#
#
# class HtmlLoggerTests(_LoggerFixture):
#     def test_content_and_nested_contexts_have_distinct_returns(self) -> None:
#         self.assertIsNone(
#             self.logger.tag("div", content="<b>raw</b>", class_="my-class")
#         )
#         with self.logger.tag("div", style="display: flex") as div:
#             self.assertIsInstance(div, HtmlLogger)
#             self.assertEqual(div.path, self.logger.path)
#             with div.span(class_="nested") as span:
#                 span.text("A < B & C")
#                 # Read while both scopes are still open.
#                 self.parsed()
#             div.p(content="after")
#         self.assertTrue(div.closed)
#         self.assertIn('<div class="my-class"><b>raw</b></div>', self.html())
#         self.assertIn(
#             '<div style="display: flex"><span class="nested">'
#             "A &lt; B &amp; C</span><p>after</p></div>",
#             self.html(),
#         )
#         self.parsed()
#
#     def test_explicit_empty_content_is_not_a_context(self) -> None:
#         self.assertIsNone(self.logger.span(content=""))
#         self.assertIsNone(self.logger.span(content=None))
#         self.assertIn("<span></span><span></span>", self.html())
#
#     def test_attributes_are_escaped_and_python_aliases_are_supported(self) -> None:
#         self.logger.tag(
#             "input",
#             content=None,
#             name="field",
#             type="checkbox",
#             checked=True,
#             disabled=False,
#             title='"<&>',
#             data_test_id="ä",
#             hidden=None,
#         )
#         self.logger.div(content="ok", **{"class": "literal", "aria-label": "name"})
#         parser = self.parsed()
#         input_attrs = next(attrs for tag, attrs in parser.elements if tag == "input")
#         self.assertEqual(input_attrs["name"], "field")
#         self.assertEqual(input_attrs["title"], '"<&>')
#         self.assertEqual(input_attrs["data-test-id"], "ä")
#         self.assertIn("checked", input_attrs)
#         self.assertNotIn("disabled", input_attrs)
#         self.assertNotIn("hidden", input_attrs)
#         self.assertIn("&quot;&lt;&amp;&gt;", self.html())
#
#     def test_plain_text_is_safe_and_raw_content_is_preserved(self) -> None:
#         self.logger.text("<script>&\n")
#         self.logger.raw("<strong>trusted</strong>")
#         self.logger.text(42)
#         self.assertIn("&lt;script&gt;&amp;\n<strong>trusted</strong>42", self.html())
#         self.assertIn("<title>Log &lt;&amp;&gt; &quot;ä&quot;</title>", self.html())
#
#     def test_contexts_close_when_an_exception_propagates(self) -> None:
#         with self.assertRaisesRegex(ValueError, "original error"):
#             with self.logger.div() as div:
#                 with div.span() as span:
#                     span.text("before failure")
#                     raise ValueError("original error")
#         self.logger.p(content="recovered")
#         self.assertIn(
#             "<div><span>before failure</span></div><p>recovered</p>", self.html()
#         )
#         self.parsed()
#
#     def test_reopen_appends_inside_one_document(self) -> None:
#         self.logger.p(content="first")
#         self.logger.close()
#         with HtmlLogger(self.path, title="ignored on reopen") as reopened:
#             reopened.p(content="second")
#         html = self.html()
#         self.assertLess(html.index("first"), html.index("second"))
#         self.assertEqual(html.count("<!DOCTYPE html>"), 1)
#         self.assertEqual(html.count("</body>"), 1)
#         self.assertEqual(html.count("</html>"), 1)
#         self.assertNotIn("ignored on reopen", html)
#         self.parsed()
#
#     def test_root_close_finishes_open_scopes_and_is_idempotent(self) -> None:
#         div = self.logger.div()
#         span = div.span()
#         span.text("remaining")
#         self.logger.close()
#         self.logger.close()
#         span.close()
#         div.close()
#         self.assertTrue(span.closed)
#         self.assertTrue(div.closed)
#         self.parsed()
#         with self.assertRaisesRegex(RuntimeError, "closed"):
#             span.text("too late")
#         with HtmlLogger(self.path) as reopened:
#             reopened.p(content="after close")
#         self.parsed()
#
#     def test_out_of_order_operations_fail_without_corrupting_html(self) -> None:
#         with self.logger.div() as div:
#             with div.span() as span:
#                 for operation in (self.logger.text, div.text):
#                     with self.assertRaisesRegex(RuntimeError, "innermost"):
#                         operation("wrong scope")
#                 with self.assertRaisesRegex(RuntimeError, "innermost"):
#                     div.close()
#                 span.text("right scope")
#         self.assertNotIn("wrong scope", self.html())
#         self.parsed()
#
#     def test_void_shortcuts_and_invalid_names(self) -> None:
#         self.assertIsNone(self.logger.br())
#         self.assertIsNone(self.logger.hr(class_="separator"))
#         before = self.html()
#         for name in ('div onclick="bad"', "</div>", "", "a b"):
#             with self.assertRaises(ValueError):
#                 self.logger.tag(name, content="no")
#         with self.assertRaises(ValueError):
#             self.logger.tag("br")
#         with self.assertRaises(ValueError):
#             self.logger.tag("img", content="not allowed")
#         with self.assertRaises(ValueError):
#             self.logger.div(content="no", **{"class": "a", "class_": "b"})
#         with self.assertRaises(ValueError):
#             self.logger.div(content="no", **{'bad"name': "bad"})
#         self.assertEqual(before, self.html())
#         self.parsed()
#
#     def test_unrelated_file_is_never_overwritten(self) -> None:
#         other = self.path.with_name("other.html")
#         original = b"<html><body>Existing user document</body></html>"
#         other.write_bytes(original)
#         with self.assertRaisesRegex(ValueError, "not an HtmlLogger document"):
#             HtmlLogger(other)
#         self.assertEqual(other.read_bytes(), original)
#
#
# class HtmlLoggerImageSavingTests(_LoggerFixture):
#     """Test HtmlLogger's image-file handling without OpenCV or NumPy."""
#
#     def test_saved_path_and_existing_files_are_preserved(self) -> None:
#         original = self.path.parent / "log_image_000001.png"
#         original.write_bytes(b"existing image")
#         path = self.logger._document.save_image(b"new image", ".png")
#         self.assertIsInstance(path, Path)
#         self.assertEqual(path, self.path.parent / "log_image_000002.png")
#         self.assertEqual(path.read_bytes(), b"new image")
#         self.assertEqual(original.read_bytes(), b"existing image")
#
#     def test_partial_files_are_removed_and_write_errors_propagate(self) -> None:
#         original_open = Path.open
#         for error_type in (OSError, FileExistsError):
#             with self.subTest(error_type=error_type):
#                 failure = error_type("write failed after file creation")
#
#                 @contextmanager
#                 def failing_file(path):
#                     with original_open(path, "xb") as stream:
#                         def fail(data):
#                             stream.write(data[:3])
#                             raise failure
#
#                         yield SimpleNamespace(write=fail)
#
#                 def open_file(path, mode="r", *args, **kwargs):
#                     if mode == "xb":
#                         return failing_file(path)
#                     return original_open(path, mode, *args, **kwargs)
#
#                 with patch.object(Path, "open", new=open_file):
#                     with self.assertRaises(error_type) as raised:
#                         self.logger._document.save_image(
#                             b"partially written image", ".png"
#                         )
#                 self.assertIs(raised.exception, failure)
#                 self.assertEqual(list(self.path.parent.glob("*.png")), [])
#
#     def test_open_errors_do_not_remove_existing_images(self) -> None:
#         original_open = Path.open
#         existing = self.path.parent / "log_image_000001.png"
#         existing.write_bytes(b"keep this image")
#
#         def open_file(path, mode="r", *args, **kwargs):
#             if mode == "xb" and path != existing:
#                 raise PermissionError("cannot create next image")
#             return original_open(path, mode, *args, **kwargs)
#
#         with patch.object(Path, "open", new=open_file):
#             with self.assertRaises(PermissionError):
#                 self.logger._document.save_image(b"new image", ".png")
#         self.assertEqual(existing.read_bytes(), b"keep this image")
#         self.assertEqual(list(self.path.parent.glob("*.png")), [existing])
#
#
# @unittest.skipUnless(
#     importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
#     "Image tests require OpenCV and NumPy",
# )
# class ImageTests(_LoggerFixture):
#     def setUp(self) -> None:
#         super().setUp()
#         import numpy as np
#
#         self.np = np
#         self.image = np.zeros((8, 12, 3), dtype=np.uint8)
#         self.image[:, :] = (10, 80, 240)  # OpenCV BGR
#
#     def test_png_roundtrip_and_link_in_a_nested_scope(self) -> None:
#         import cv2
#
#         with self.logger.div() as div:
#             self.assertIsNone(
#                 div.img(
#                     self.image,
#                     alt='image <&> "ä"',
#                     width=120,
#                     params=[cv2.IMWRITE_PNG_COMPRESSION, 9],
#                     link_attrs={"class_": "open-image"},
#                 )
#             )
#             parser = self.parsed()
#         image_attrs = next(attrs for tag, attrs in parser.elements if tag == "img")
#         link_attrs = next(attrs for tag, attrs in parser.elements if tag == "a")
#         self.assertEqual(link_attrs["target"], "_blank")
#         self.assertEqual(link_attrs["class"], "open-image")
#         self.assertEqual(link_attrs["href"], image_attrs["src"])
#         self.assertEqual(image_attrs["alt"], 'image <&> "ä"')
#         image_path = self.path.parent / unquote(image_attrs["src"] or "")
#         self.assertEqual(image_path.name, "log_image_000001.png")
#         decoded = cv2.imdecode(
#             self.np.frombuffer(image_path.read_bytes(), dtype=self.np.uint8),
#             cv2.IMREAD_UNCHANGED,
#         )
#         self.np.testing.assert_array_equal(decoded, self.image)
#
#     def test_numbering_survives_reopen_and_existing_files(self) -> None:
#         existing = self.path.parent / "log_image_000007.jpg"
#         existing.write_bytes(b"keep this")
#         self.logger.close()
#         with HtmlLogger(self.path) as reopened:
#             reopened.img(self.image)
#         with HtmlLogger(self.path) as reopened:
#             reopened.img(self.image, extension="PNG")
#         self.assertEqual(existing.read_bytes(), b"keep this")
#         self.assertTrue((self.path.parent / "log_image_000008.png").is_file())
#         self.assertTrue((self.path.parent / "log_image_000009.png").is_file())
#         self.parsed()
#
#     def test_newly_created_image_is_not_overwritten(self) -> None:
#         existing = self.path.parent / "log_image_000001.png"
#         existing.write_bytes(b"created after logger initialization")
#         self.logger.img(self.image)
#         self.assertEqual(existing.read_bytes(), b"created after logger initialization")
#         self.assertTrue((self.path.parent / "log_image_000002.png").is_file())
#
#     def test_unicode_and_special_characters_in_paths(self) -> None:
#         path = self.path.parent / "Umläute & # [test]" / "Größe #1.html"
#         with HtmlLogger(path) as logger:
#             logger.img(self.image)
#         parser = DocumentParser()
#         parser.feed(path.read_text(encoding="utf-8"))
#         attrs = next(attrs for tag, attrs in parser.elements if tag == "img")
#         url = attrs["src"] or ""
#         self.assertNotIn("#", url)
#         self.assertNotIn(" ", url)
#         self.assertTrue((path.parent / unquote(url)).is_file())
#
#     def test_grayscale_and_bgra_are_preserved(self) -> None:
#         import cv2
#
#         for number, image in enumerate(
#                 (
#                         self.np.arange(24, dtype=self.np.uint16).reshape(4, 6),
#                         self.np.full((4, 6, 4), (20, 80, 140, 100), dtype=self.np.uint8),
#                 ),
#                 start=1,
#         ):
#             self.logger.img(image)
#             path = self.path.parent / f"log_image_{number:06d}.png"
#             decoded = cv2.imdecode(
#                 self.np.frombuffer(path.read_bytes(), dtype=self.np.uint8),
#                 cv2.IMREAD_UNCHANGED,
#             )
#             self.np.testing.assert_array_equal(decoded, image)
#
#     def test_image_failures_leave_html_and_files_unchanged(self) -> None:
#         before = self.html()
#         before_files = set(self.path.parent.iterdir())
#         cases = (
#             (self.np.zeros((0, 2), dtype=self.np.uint8), {}),
#             (self.np.zeros((2, 2, 2), dtype=self.np.uint8), {}),
#             (self.image, {"extension": "../png"}),
#             (self.image, {"extension": ".not-a-codec"}),
#             (self.image, {"extension": ".noencoder"}),
#             (self.image, {"src": "other.png"}),
#             (self.image, {"link_attrs": {"target": "_self"}}),
#         )
#         for image, attrs in cases:
#             with self.assertRaises(ValueError):
#                 self.logger.img(image, **attrs)
#         with self.assertRaises(TypeError):
#             self.logger.img([[1, 2]])
#         with patch(
#                 "cv2.imencode", return_value=(False, self.np.array([], dtype=self.np.uint8))
#         ):
#             with self.assertRaises(ValueError):
#                 self.logger.img(self.image)
#         self.assertEqual(self.html(), before)
#         self.assertEqual(set(self.path.parent.iterdir()), before_files)
#
#
# if __name__ == "__main__":
#     unittest.main()
