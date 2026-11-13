"""Synthetic tests for region routing and source-coordinate restoration."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import types
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from src.local_layout import LocalLayout, validate_layout
from src.paddle_layout_ocr import LayoutRegionOCR, create_recognizer
from src.paddle_ocr_trial import file_hash, model_settings, run_trial


def polygon(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def layout_data():
    return {"image_size": [1000, 1000], "predictions": [
        {"label": "Title", "confidence": .9, "bbox_xyxy": [100, 100, 700, 200]},
        {"label": "Content", "confidence": .95, "bbox_xyxy": [50, 150, 750, 950]},
        {"label": "Notes", "confidence": .8, "bbox_xyxy": [700, 350, 900, 700]},
    ]}


class Result:
    def __init__(self, value):
        self.value = value

    def save_to_json(self, path):
        Path(path).write_text(json.dumps(self.value), encoding="utf-8")


class Layout:
    def __init__(self, document=None):
        self.document = layout_data() if document is None else document

    def describe(self):
        return {"model": "synthetic test layout"}

    def __call__(self, image, folder):
        return deepcopy(self.document)


class Recognition:
    def __init__(self, text="unchanged title prediction"):
        self.text = text
        self.inputs = []

    def predict(self, path):
        self.inputs.append(Path(path))
        return [Result({"rec_text": self.text, "rec_score": .8})]


class OCR:
    def __init__(self):
        self.inputs = []
        self.title_was_masked = self.notes_were_masked = False

    def predict(self, path):
        self.inputs.append(Path(path))
        with Image.open(path) as crop:
            self.title_was_masked = crop.getpixel((100, 20)) == (255, 255, 255)
            self.notes_were_masked = crop.getpixel((670, 250)) == (255, 255, 255)
        # One valid column, one spurious residual in Notes, one in Title.
        boxes = [polygon(100, 100, 130, 500), polygon(660, 220, 690, 400), polygon(100, 10, 150, 30)]
        return [Result({"dt_polys": boxes, "rec_polys": boxes,
                        "rec_texts": ["body", "note", "duplicate title"], "rec_scores": [.8, .9, .7]})]


def source_image(path):
    image = Image.new("RGB", (1000, 1000), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 700, 200), fill="red")
    draw.rectangle((700, 350, 900, 700), fill="blue")
    image.save(path)


class PaddleLayoutTests(unittest.TestCase):
    def test_title_uses_the_selected_gpu_instead_of_falling_back_to_cpu(self):
        factory = Mock()
        fake_package = types.ModuleType("paddleocr")
        fake_package.TextRecognition = factory
        with patch.dict("sys.modules", {"paddleocr": fake_package}):
            create_recognizer(model_settings(device="gpu:0"))
        self.assertEqual(factory.call_args.kwargs["device"], "gpu:0")

    def test_title_is_one_box_notes_are_not_ocrd_and_body_coordinates_are_restored(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "image.png"
            source_image(source)
            before = file_hash(source)
            folder = root / "output"
            folder.mkdir()
            recognizer, model = Recognition(), OCR()
            processor = LayoutRegionOCR(Layout(), recognizer_factory=lambda _: recognizer, progress=lambda _: None)
            document = processor(source, folder, "bia_99.001", [1000, 1000], model, model_settings())
            self.assertEqual(document["header_box_count"], 1)
            self.assertEqual(document["recognized_box_count"], 2)
            self.assertEqual(document["notes_region_count"], 1)
            self.assertEqual(len(recognizer.inputs), 1)
            self.assertEqual(len(model.inputs), 1)
            self.assertTrue(model.title_was_masked)
            self.assertTrue(model.notes_were_masked)
            title, body = document["boxes"]
            self.assertEqual(title["bbox"], polygon(100, 100, 700, 200))
            self.assertEqual(title["sino_nom_ocr"], recognizer.text)
            self.assertEqual(title["region"], "header")
            self.assertEqual(body["bbox"], polygon(150, 250, 180, 650))
            self.assertEqual(body["bbox_in_crop"], polygon(100, 100, 130, 500))
            self.assertEqual(body["region"], "body")
            excluded = json.loads((folder / "excluded_boxes.json").read_text())
            self.assertEqual(len(excluded["boxes"]), 4)  # detection + recognition
            self.assertEqual(file_hash(source), before)
            self.assertIsNone(title["reading_order"])
            self.assertIsNone(title["sino_nom"])

    def test_absent_content_stops_before_calling_recognition_or_ocr(self):
        data = layout_data()
        data["predictions"] = [p for p in data["predictions"] if p["label"] != "Content"]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "image.png"
            source_image(source)
            model, factory = Mock(), Mock()
            processor = LayoutRegionOCR(Layout(data), recognizer_factory=factory, progress=lambda _: None)
            with self.assertRaises(ValueError):
                processor(source, root, "test", [1000, 1000], model, model_settings())
            model.predict.assert_not_called()
            factory.assert_not_called()

    def test_empty_title_retains_layout_geometry_and_reports_missing_text(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "image.png"
            source_image(source)
            processor = LayoutRegionOCR(Layout(), recognizer_factory=lambda _: Recognition(""), progress=lambda _: None)
            out = processor(source, root, "test", [1000, 1000], OCR(), model_settings())
            self.assertEqual(out["header_box_count"], 1)
            self.assertEqual(out["header_recognized_count"], 0)
            self.assertIn("title_without_recognized_text", out["boxes"][0]["review_flags"])

    def test_missing_title_does_not_invent_a_header(self):
        data = layout_data()
        data["predictions"] = [p for p in data["predictions"] if p["label"] != "Title"]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "image.png"
            source_image(source)
            factory = Mock()
            out = LayoutRegionOCR(Layout(data), recognizer_factory=factory, progress=lambda _: None)(
                source, root, "test", [1000, 1000], OCR(), model_settings())
            self.assertEqual(out["header_box_count"], 0)
            self.assertIn("title_not_found_by_layout", out["warnings"])
            factory.assert_not_called()

    def test_pipeline_can_save_region_outputs_without_changing_baseline_codepath(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "image.png"
            source_image(source)
            processor = LayoutRegionOCR(Layout(), recognizer_factory=lambda _: Recognition(), progress=lambda _: None)
            report = run_trial([source], root / "results", model_settings(),
                               model_factory=lambda _: OCR(), region_processor=processor, progress=lambda _: None)
            self.assertEqual(report["summary"]["saved_for_review"], 1)
            row = report["files"][0]
            self.assertEqual(row["header_box_count"], 1)
            self.assertEqual(row["notes_region_count"], 1)
            self.assertTrue(Path(row["preview"]).is_file())
            self.assertTrue((Path(row["output_dir"]) / "regions/title_001/raw_result.json").is_file())
            self.assertTrue((Path(row["output_dir"]) / "regions/content_001/raw_result.json").is_file())

    def test_bad_layout_and_mismatched_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_layout(layout_data(), [100, 100])
        data = layout_data()
        data["predictions"][0]["bbox_xyxy"][0] = -1
        with self.assertRaises(ValueError):
            validate_layout(data, [1000, 1000])

    def test_layout_bridge_uses_separate_python_without_shell_or_autoinstall(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            python, model = root / "python.exe", root / "best.pt"
            python.write_bytes(b"synthetic")
            model.write_bytes(b"synthetic checkpoint not executed")
            image = root / "image.png"
            source_image(image)
            def worker(command, **kwargs):
                self.assertEqual(command[0], str(python))
                self.assertFalse(kwargs.get("shell", False))
                self.assertEqual(kwargs["env"]["YOLO_AUTOINSTALL"], "False")
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                self.assertEqual(command[command.index("--device") + 1], "cuda:0")
                output = Path(command[command.index("--output") + 1])
                data = layout_data()
                data.update(image_sha256=file_hash(image), model_sha256=file_hash(model))
                output.write_text(json.dumps(data), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "LAYOUT_REGIONS: 3", "")
            with patch("src.local_layout.subprocess.run", side_effect=worker):
                bridge = LocalLayout(python, model, device="cuda:0")
                out = bridge(image, root)
            self.assertEqual(len(out["predictions"]), 3)
            self.assertEqual(bridge.describe()["device"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
