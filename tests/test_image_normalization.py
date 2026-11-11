"""Synthetic image tests only: no Lab documents or OCR models."""

from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import BytesIO, StringIO
import json
import unittest
from unittest.mock import patch

from PIL import Image
import pymupdf

from run_image_pipeline import main as cli_main
from src.image_ocr import (
    _write_new_files, load_source_image, normalize_image, normalize_single_input,
    resize_page_image,
)


class ImageNormalizationTests(unittest.TestCase):
    def test_native_jpeg_copy_ignores_dataset_exports_and_preserves_encoded_bytes(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'native.jpg'
            Image.new('RGB', (1800, 2700), 'white').save(source)
            original = source.read_bytes()
            legacy = root / 'data/dataset/book_a/images/bia_test.001.jpg'
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b'old export, unrelated to processing')
            result = normalize_single_input(source, 'bia_test', book_id='book_a', app_root=root)
            self.assertEqual(result['image_id'], 'bia_test.001')
            self.assertEqual(result['image_size'], [1800, 2700])
            self.assertEqual(result['coordinate_space'], 'prepared_image_pixels')
            self.assertEqual(result['processing']['encoding'], 'copy_jpeg')
            self.assertEqual((root / result['output_jpg']).read_bytes(), original)
            self.assertEqual(legacy.read_bytes(), b'old export, unrelated to processing')

    def test_pdf_embedded_image_above_1024_keeps_native_dimensions(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'native.pdf'
            stream = BytesIO()
            Image.new('RGB', (1500, 2100), 'white').save(stream, format='JPEG')
            with pymupdf.open() as document:
                page = document.new_page(width=150, height=210)
                xref = page.insert_image(page.rect, stream=stream.getvalue())
                document.xref_set_key(xref, 'ColorSpace', '/DeviceRGB')
                document.save(source)
            result = normalize_single_input(source, 'bia_test', book_id='book_a', app_root=root)
            self.assertEqual(result['load_method'], 'extract')
            self.assertEqual(result['image_size'], [1500, 2100])
            self.assertEqual(result['resize_scale'], {'x': 1.0, 'y': 1.0})

    def test_shrink_portrait_and_keep_aspect_ratio(self):
        source = Image.new("RGB", (2560, 3544), "white")
        result = normalize_image(source)
        self.assertEqual(result.size, (740, 1024))
        self.assertLessEqual(abs(result.width - source.width * 1024 / source.height), 0.5)
        self.assertEqual(source.size, (2560, 3544))

    def test_landscape_and_very_thin_image(self):
        for source_size, expected in [((3000, 1500), (1024, 512)), ((1, 3000), (1, 1024))]:
            with self.subTest(size=source_size):
                self.assertEqual(normalize_image(Image.new("RGB", source_size)).size, expected)

    def test_small_image_not_upscaled_or_rotated(self):
        source = Image.new("RGB", (30, 70), "white")
        for xy, color in [((0, 0), "red"), ((29, 0), "green"),
                          ((0, 69), "blue"), ((29, 69), "black")]:
            source.paste(color, (*xy, xy[0] + 1, xy[1] + 1))
        result = normalize_image(source)
        self.assertEqual(result.size, source.size)
        self.assertEqual(result.tobytes(), source.tobytes())

    def test_convert_grayscale_palette_and_cmyk_to_rgb(self):
        for mode in ("1", "L", "P", "CMYK"):
            with self.subTest(mode=mode):
                source = Image.new(mode, (20, 40))
                result = normalize_image(source)
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.size, source.size)

    def test_transparency_composited_over_white(self):
        image = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
        image.putpixel((0, 0), (255, 0, 0, 255))
        result = normalize_image(image)
        self.assertEqual(result.getpixel((1, 1)), (255, 255, 255))
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))

    def test_invalid_maximum_rejected(self):
        for value in (0, -1, True, 10.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_image(Image.new("RGB", (2, 2)), value)

    def test_raster_function_accepts_readme_arguments(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.png"
            Image.new("L", (80, 40)).save(path)
            result = resize_page_image(image_path=str(path), max_dim=40)
            self.assertEqual(result.mode, "RGB")
            self.assertEqual(result.size, (40, 20))

    def test_exclusive_writer_never_overwrites_existing_file(self):
        with TemporaryDirectory() as temp:
            existing = Path(temp) / "existing.jpg"
            other = Path(temp) / "other.json"
            existing.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                _write_new_files({other: b"metadata", existing: b"replacement"})
            self.assertEqual(existing.read_bytes(), b"original")
            self.assertFalse(other.exists())

    def test_book_output_guard_preserves_existing_pilot_without_reading_pdf(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "data/image_pipeline/book_a/bia_20367/01_prepare/bia_20367.001.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"existing pilot")
            with self.assertRaises(FileExistsError):
                normalize_single_input("missing.pdf", "bia_20367", book_id="book_a", image_number=1, app_root=root)
            self.assertEqual(image.read_bytes(), b"existing pilot")
            self.assertEqual([p for p in root.rglob("*") if p.is_file()], [image])

    def test_output_in_wrong_book_is_rejected_before_processing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "Output must be inside"):
                normalize_single_input(
                    "missing.pdf", "bia_20367", "data/dataset/book_b/images/bia_20367.001.jpg",
                    book_id="book_a", image_number=1, app_root=root,
                )
            self.assertFalse(any(root.iterdir()))

    def test_image_index_must_be_valid_and_three_digits(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for invalid in (0, -1, 1000, True, 1.5, '', '000', '0001', '1.0', '../1'):
                with self.subTest(page=invalid), self.assertRaisesRegex(ValueError, 'image_number'):
                    normalize_single_input('missing.jpg', 'bia_test', book_id='book_a',
                                           image_number=invalid, app_root=root)
            self.assertFalse(any(root.iterdir()))

    def test_two_images_can_share_supplied_page_without_name_collisions(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / 'first.jpg', root / 'second.jpg']
            results = []
            for source in sources:
                Image.new('RGB', (20, 40), 'white').save(source)
                results.append(normalize_single_input(source, 'bia_group', book_id='book_a',
                                                       page_number=65, app_root=root, preview=True))
            self.assertEqual([r['image_id'] for r in results], ['bia_group.001', 'bia_group.002'])
            self.assertTrue(all(r['book_page_number'] == 65 for r in results))
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with self.assertRaises(FileExistsError):
                normalize_single_input(sources[0], 'bia_group', book_id='book_a', app_root=root)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_page_based_output_filename_is_rejected_for_new_output(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'Output filename must be bia_test.001.jpg'):
                normalize_single_input('missing.jpg', 'bia_test',
                                       'data/dataset/book_a/images/book_a.025.jpg',
                                       book_id='book_a', page_number=25, app_root=root)
            self.assertFalse(any(root.iterdir()))

    def test_jpeg_pipeline_writes_rgb_metadata_and_local_image_preview(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "synthetic.JPEG"
            Image.new("L", (2000, 1000), 128).save(source)
            original = source.read_bytes()
            metadata = normalize_single_input(
                source, "bia_test", book_id="book_a", page_number=25, app_root=root, preview=True,
            )
            self.assertEqual(metadata["source_type"], "jpeg")
            self.assertEqual(metadata["image_id"], "bia_test.001")
            self.assertEqual(metadata["page_number"], 25)
            self.assertEqual(metadata["book_page_number"], 25)
            self.assertEqual(metadata["image_number"], "001")
            self.assertEqual(metadata["page_number_source"], 'provided_explicit')
            self.assertEqual(metadata["output_jpg"], "data/image_pipeline/book_a/bia_test/01_prepare/bia_test.001.jpg")
            self.assertEqual(metadata["metadata_path"], "data/image_pipeline/book_a/bia_test/01_prepare/bia_test.001.json")
            self.assertEqual(metadata["load_method"], "direct_image")
            self.assertIsNone(metadata["source_page"])
            self.assertEqual(metadata["source_sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(metadata["source_unchanged"])
            for key in ("input_pdf", "source_pdf_page", "page_count", "embedded_images"):
                self.assertNotIn(key, metadata)
            with Image.open(root / metadata["output_jpg"]) as output:
                self.assertEqual(output.size, (2000, 1000))
                self.assertEqual(output.mode, "RGB")
                self.assertEqual(output.format, "JPEG")
            saved = json.loads((root / metadata["metadata_path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved, metadata)
            html = (root / metadata["preview_path"]).read_text(encoding="utf-8")
            self.assertIn('alt="JPG gốc"', html)
            self.assertNotIn("<iframe", html)
            self.assertNotIn("https://", html)
            self.assertIn("bia_test.001.jpg", html)
            self.assertIsNone(metadata["processing"]["render_dpi"])
            self.assertEqual(metadata['processing']['resolution_policy'], 'native')
            self.assertIsNone(metadata['processing']['max_dim'])
            self.assertEqual(metadata['resize_scale'], {'x': 1.0, 'y': 1.0})

    def test_jpeg_exif_orientation_is_respected_without_upscaling(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "oriented.jpg"
            image = Image.new("RGB", (60, 40), "red")
            image.paste("green", (30, 0, 60, 20))
            image.paste("blue", (0, 20, 30, 40))
            image.paste("yellow", (30, 20, 60, 40))
            exif = Image.Exif()
            exif[274] = 6
            image.save(source, quality=100, subsampling=0, exif=exif)
            original = source.read_bytes()
            loaded, info = load_source_image(source)
            self.assertEqual(loaded.size, (40, 60))
            red, green, blue = loaded.getpixel((5, 5))
            self.assertGreater(blue, 200)
            self.assertLess(max(red, green), 30)
            metadata = normalize_single_input(source, "bia_test", book_id="book_a", image_number=1, app_root=root)
            self.assertEqual(metadata["encoded_image_size"], [60, 40])
            self.assertEqual(metadata["image_size_before_resize"], [40, 60])
            self.assertEqual(metadata["image_size_after_resize"], [40, 60])
            self.assertEqual(info["exif_orientation"], 6)
            self.assertTrue(metadata["exif_orientation_applied"])
            with Image.open(root / metadata["output_jpg"]) as output:
                self.assertEqual(output.size, (40, 60))
                self.assertNotIn(274, output.getexif())
            self.assertEqual(source.read_bytes(), original)

    def test_png_disguised_as_jpeg_is_rejected_without_outputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "wrong.jpg"
            Image.new("RGB", (20, 40)).save(source, format="PNG")
            with self.assertRaisesRegex(ValueError, "content is not JPEG"):
                normalize_single_input(source, "bia_test", book_id="book_a", image_number=1, app_root=root)
            self.assertFalse((root / "data").exists())

    def test_broken_jpeg_is_rejected_without_outputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "broken.jpg"
            source.write_bytes(b"not a JPEG")
            with self.assertRaises(OSError):
                normalize_single_input(source, "bia_test", book_id="book_a", image_number=1, app_root=root)
            self.assertFalse((root / "data").exists())
            self.assertEqual(source.read_bytes(), b"not a JPEG")

    def test_synthetic_single_image_pdf_still_extracts_with_legacy_aliases(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "synthetic.pdf"
            image_bytes = BytesIO()
            Image.new("RGB", (200, 360), "blue").save(image_bytes, format="JPEG")
            with pymupdf.open() as document:
                page = document.new_page(width=100, height=180)
                xref = page.insert_image(page.rect, stream=image_bytes.getvalue())
                # Test the plain DeviceRGB extraction case; ICC PDFs render conservatively.
                document.xref_set_key(xref, "ColorSpace", "/DeviceRGB")
                document.save(source)
            original = source.read_bytes()
            metadata = normalize_single_input(source, "bia_test", book_id="book_a", image_number=1, app_root=root)
            self.assertEqual(metadata["source_type"], "pdf")
            self.assertEqual(metadata["load_method"], "extract")
            self.assertEqual(metadata["source_page"], 1)
            self.assertEqual(metadata["page_count"], 1)
            self.assertEqual(metadata["input_pdf"], metadata["source_path"])
            self.assertTrue(metadata["input_pdf_unchanged"])
            self.assertEqual(source.read_bytes(), original)

    def test_synthetic_pdf_with_two_images_uses_render_fallback(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "composite.pdf"
            image_bytes = BytesIO()
            Image.new("RGB", (40, 40), "red").save(image_bytes, format="JPEG")
            with pymupdf.open() as document:
                page = document.new_page(width=100, height=200)
                page.insert_image(pymupdf.Rect(0, 0, 100, 100), stream=image_bytes.getvalue())
                page.insert_image(pymupdf.Rect(0, 100, 100, 200), stream=image_bytes.getvalue())
                document.save(source)
            metadata = normalize_single_input(
                source, "bia_test", book_id="book_a", image_number=1, app_root=root, render_dpi=72,
            )
            self.assertEqual(metadata["load_method"], "render")
            self.assertEqual(metadata["processing"]["render_dpi"], 72)
            self.assertEqual(metadata["image_size_after_resize"], [100, 200])

    def test_synthetic_multi_page_pdf_is_rejected_without_outputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "two_pages.pdf"
            with pymupdf.open() as document:
                document.new_page()
                document.new_page()
                document.save(source)
            with self.assertRaisesRegex(ValueError, "exactly one PDF page"):
                normalize_single_input(source, "bia_test", book_id="book_a", image_number=1, app_root=root)
            self.assertFalse((root / "data").exists())

    def test_cli_input_accepts_pdf_and_jpeg(self):
        summary = {
            "book_id": "book_a", "stele_id": "bia_test", "source_type": "pdf",
            "image_id": "bia_test.001", "image_number": "001", "page_number": 25,
            "load_method": "extract", "source_path": "synthetic.pdf",
            "image_size_before_resize": [100, 200], "image_size_after_resize": [100, 200],
            "output_jpg": "output.jpg", "metadata_path": "metadata.json", "source_unchanged": True,
        }
        for source in ("synthetic.pdf", "synthetic.jpg"):
            with self.subTest(source=source), patch("sys.argv", [
                "run_image_pipeline.py", "--normalize-only", "--input", source, "--book-id", "book_a",
                "--stele-id", "bia_test", "--page-number", "25",
            ]), patch("run_image_pipeline._normalize_selected_input", return_value=summary) as normalize:
                with redirect_stdout(StringIO()):
                    self.assertEqual(cli_main(), 0)
                self.assertEqual(normalize.call_args.args[:2], (source, "bia_test"))
                self.assertEqual(normalize.call_args.kwargs["page_number"], 25)

    def test_cli_without_supplied_page_passes_none(self):
        with patch('sys.argv', ['run_image_pipeline.py', '--normalize-only', '--input', 'synthetic.jpg',
                                '--book-id', 'book_a', '--stele-id', 'bia_test']), \
                patch('run_image_pipeline._normalize_selected_input', side_effect=ValueError('synthetic stop')) as normalize:
            with redirect_stderr(StringIO()):
                self.assertEqual(cli_main(), 1)
            normalize.assert_called_once()
            self.assertIsNone(normalize.call_args.kwargs['page_number'])

    def test_auto_indices_are_saved_per_source_and_separate_per_book(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / f'synthetic_{n}.jpg' for n in range(2)]
            for source in sources:
                Image.new('RGB', (20, 40), 'white').save(source)
            results = [normalize_single_input(source, 'bia_group', book_id='book_a', app_root=root)
                       for source in sources]
            self.assertEqual([r['image_id'] for r in results], ['bia_group.001', 'bia_group.002'])
            self.assertTrue(all(r['book_page_number'] is None for r in results))
            self.assertTrue(all(r['page_number_source'] is None for r in results))
            snapshot = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch('src.image_ocr.load_source_image') as load:
                with self.assertRaises(FileExistsError):
                    normalize_single_input(sources[0], 'bia_group', book_id='book_a', app_root=root)
                load.assert_not_called()
            self.assertEqual(snapshot, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})
            other = normalize_single_input(sources[0], 'bia_group', book_id='book_b', app_root=root)
            self.assertEqual(other['image_id'], 'bia_group.001')

    def test_auto_index_reserves_existing_jpg_and_preview_without_metadata(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            jpg = root / 'data/image_pipeline/book_a/bia_test/01_prepare/bia_test.003.jpg'
            preview = root / 'data/image_pipeline/book_a/bia_test/bia_test.004.preview.html'
            for path in (jpg, preview):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'existing')
            source = root / 'synthetic.jpg'
            Image.new('RGB', (20, 40)).save(source)
            result = normalize_single_input(source, 'bia_test', book_id='book_a', app_root=root)
            self.assertEqual(result['image_id'], 'bia_test.005')
            for path in (jpg, preview):
                self.assertEqual(path.read_bytes(), b'existing')

    def test_auto_index_stops_at_limit_without_loading_source(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            jpg = root / 'data/image_pipeline/book_a/bia_test/01_prepare/bia_test.999.jpg'
            jpg.parent.mkdir(parents=True)
            jpg.write_bytes(b'existing')
            with patch('src.image_ocr.load_source_image') as load:
                with self.assertRaisesRegex(ValueError, 'image_number'):
                    normalize_single_input('missing.jpg', 'bia_test', book_id='book_a', app_root=root)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
