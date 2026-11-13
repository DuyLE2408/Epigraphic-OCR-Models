"""Isolated local-OCR contract tests: synthetic files and an injected fake model."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import pymupdf

from src.paddle_ocr_trial import convert_result, file_hash, model_settings, run_trial
from run_paddle_ocr_trial import main


POLYGON = [[2, 3], [18, 3], [18, 35], [2, 35]]
OTHER = [[30, 3], [45, 3], [45, 35], [30, 35]]


def result_data():
    # Detection has an extra box, recognition has a different subset/order.
    return {"dt_polys": [POLYGON, OTHER], "dt_scores": [.9, .8],
            "rec_polys": [OTHER], "rec_texts": ["test prediction"], "rec_scores": [.4],
            "text_det_params": {"limit_side_len": 64, "limit_type": "min"},
            "text_rec_score_thresh": 0.0}


class FakeResult:
    def __init__(self, data):
        self.data = data

    def save_to_json(self, path):
        Path(path).write_text(json.dumps(self.data), encoding="utf-8")


class FakeModel:
    def __init__(self, data=None):
        self.data = result_data() if data is None else data
        self.inputs = []

    def predict(self, path):
        self.inputs.append(Path(path))
        return [FakeResult(self.data)]


def run(sources, root, model=None, **kwargs):
    return run_trial(sources, root, model_settings(), model_factory=lambda settings: model or FakeModel(),
                     progress=lambda message: None, **kwargs)


class PaddleTrialTests(unittest.TestCase):
    def test_settings_are_v6_cpu_and_no_layout_or_correction_models(self):
        settings = model_settings()
        self.assertEqual(settings["ocr_version"], "PP-OCRv6")
        self.assertEqual(settings["lang"], "chinese_cht")
        self.assertEqual(settings["text_detection_model_name"], "PP-OCRv6_medium_det")
        self.assertEqual(settings["text_recognition_model_name"], "PP-OCRv6_medium_rec")
        self.assertFalse(settings["use_doc_unwarping"])
        self.assertFalse(settings["use_doc_orientation_classify"])
        self.assertFalse(settings["use_textline_orientation"])
        self.assertEqual(settings["text_rec_score_thresh"], 0.0)
        self.assertNotIn("text_det_limit_side_len", settings)
        with self.assertRaises(ValueError):
            model_settings(cpu_threads=0)

    def test_recognition_text_is_paired_with_rec_polys_not_dt_polys(self):
        raw = result_data()
        before = deepcopy(raw)
        out = convert_result({"res": raw}, image_id="bia_1.001", image_size=(60, 80))
        self.assertEqual(out["detected_box_count"], 2)
        self.assertEqual(out["recognized_box_count"], 1)
        self.assertEqual(out["boxes"][0]["bbox"], OTHER)
        self.assertEqual(out["boxes"][0]["sino_nom_ocr"], "test prediction")
        self.assertIsNone(out["boxes"][0]["reading_order"])
        self.assertIsNone(out["boxes"][0]["sino_nom"])
        self.assertIsNone(out["boxes"][0]["patch_id"])
        self.assertEqual(out["boxes"][0]["region"], "unclassified")
        self.assertEqual(raw, before)

    def test_gpu_is_explicit_and_invalid_device_is_rejected(self):
        self.assertEqual(model_settings()["device"], "cpu")
        self.assertEqual(model_settings(device="gpu:0")["device"], "gpu:0")
        for device in ("cuda:0", "auto", "gpu:bad", None):
            with self.subTest(device=device), self.assertRaises(ValueError):
                model_settings(device=device)

    def test_launcher_routes_gpu_to_title_body_and_layout_with_custom_output(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("run_paddle_ocr_trial.find_spec", return_value=True), \
                 patch("src.local_layout.LocalLayout") as layout, \
                 patch("src.paddle_ocr_trial.run_trial", return_value={
                     "status": "finished", "summary": {"failed": 0, "no_recognized_text": 0},
                     "run_dir": str(root / "outputs/run")}) as trial, redirect_stdout(StringIO()):
                code = main(["--input", str(root / "image.png"), "--device", "gpu:0",
                             "--layout-device", "cuda:0", "--layout-python", str(root / "python"),
                             "--output-root", str(root / "outputs")])
            self.assertEqual(code, 0)
            self.assertEqual(layout.call_args.kwargs["device"], "cuda:0")
            self.assertEqual(trial.call_args.args[2]["device"], "gpu:0")
            self.assertEqual(trial.call_args.args[1], root / "outputs")

    def test_mismatched_recognition_and_nonfinite_geometry_are_rejected(self):
        data = result_data()
        data["rec_texts"] = []
        with self.assertRaises(ValueError):
            convert_result(data, image_id="test", image_size=(60, 80))
        data = result_data()
        data["rec_polys"] = [[[float("nan"), 0], [1, 0], [1, 1], [0, 1]]]
        with self.assertRaises(ValueError):
            convert_result(data, image_id="test", image_size=(60, 80))

    def test_out_of_bounds_is_flagged_without_silently_clipping_coordinates(self):
        data = result_data()
        data["rec_polys"][0] = [[-2, 3], [45, 3], [45, 35], [-2, 35]]
        out = convert_result(data, image_id="test", image_size=(60, 80))
        self.assertEqual(out["boxes"][0]["bbox"][0][0], -2)
        self.assertIn("coordinates_outside_prepared_image", out["boxes"][0]["review_flags"])

    def test_reruns_preserve_source_previous_results_and_native_size(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "00099.jpg"
            Image.new("RGB", (60, 80), "gray").save(source)
            before = source.read_bytes()
            first = run([source], root / "out")
            first_dir = Path(first["files"][0]["output_dir"])
            snapshots = {p.relative_to(first_dir): p.read_bytes() for p in first_dir.rglob("*") if p.is_file()}
            second = run([source], root / "out")
            self.assertNotEqual(first["run_dir"], second["run_dir"])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual((first_dir / "input.jpg").read_bytes(), before)
            self.assertEqual(snapshots, {p.relative_to(first_dir): p.read_bytes() for p in first_dir.rglob("*") if p.is_file()})
            self.assertEqual(first["files"][0]["image_size"], [60, 80])
            self.assertEqual(first["summary"]["saved_for_review"], 1)
            self.assertEqual(first["files"][0]["source_sha256"], file_hash(source))

    def test_exif_orientation_maps_to_prepared_image_coordinates(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "photo.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (60, 80)).save(source, exif=exif)
            before = file_hash(source)
            report = run([source], root / "out")
            row = report["files"][0]
            self.assertEqual(row["image_size"], [80, 60])
            metadata = json.loads((Path(row["output_dir"]) / "input_metadata.json").read_text())
            self.assertTrue(metadata["source_inspection"]["exif_orientation_applied"])
            self.assertFalse(metadata["external_resize_applied"])
            self.assertEqual(file_hash(source), before)

    def test_empty_recognition_is_reported_separately_with_detection_preserved(self):
        data = result_data()
        data.update(rec_polys=[], rec_texts=[], rec_scores=[])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "photo.png"
            Image.new("RGB", (60, 80)).save(source)
            report = run([source], root / "out", FakeModel(data))
            self.assertEqual(report["summary"]["saved_for_review"], 0)
            self.assertEqual(report["summary"]["no_recognized_text"], 1)
            self.assertEqual(report["files"][0]["detected_boxes"], 2)
            self.assertTrue((Path(report["files"][0]["output_dir"]) / "detection_preview.png").exists())

    def test_bad_file_does_not_prevent_next_selected_image_from_running(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            bad, good = root / "bad.jpg", root / "good.jpg"
            bad.write_bytes(b"not an image")
            Image.new("RGB", (60, 80)).save(good)
            report = run([bad, good], root / "out")
            self.assertEqual(report["summary"]["failed"], 1)
            self.assertEqual(report["summary"]["saved_for_review"], 1)
            self.assertTrue((Path(report["files"][0]["output_dir"]) / "error.json").exists())

    def test_model_load_failure_still_produces_a_report_without_modifying_inputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "photo.jpg"
            Image.new("RGB", (60, 80)).save(source)
            before = file_hash(source)
            factory = Mock(side_effect=ImportError("synthetic missing engine"))
            report = run_trial([source], root / "out", model_settings(), model_factory=factory, progress=lambda _: None)
            self.assertEqual(report["summary"]["failed"], 1)
            self.assertIn("synthetic missing engine", report["files"][0]["error"])
            self.assertTrue((Path(report["run_dir"]) / "report.json").exists())
            self.assertEqual(file_hash(source), before)

    def test_single_page_pdf_uses_existing_read_only_loader(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "20367.pdf"
            with pymupdf.open() as pdf:
                pdf.new_page(width=72, height=96)
                pdf.save(source)
            before = file_hash(source)
            report = run([source], root / "out", render_dpi=72)
            self.assertEqual(report["summary"]["saved_for_review"], 1)
            self.assertEqual(report["files"][0]["image_size"], [72, 96])
            self.assertEqual(file_hash(source), before)

    def test_interruption_saves_summary_and_leaves_remaining_sources_unrun(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            sources = [root / "a.jpg", root / "b.jpg"]
            for source in sources:
                Image.new("RGB", (60, 80)).save(source)
            model = FakeModel()
            model.predict = Mock(side_effect=KeyboardInterrupt())
            report = run(sources, root / "out", model)
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["summary"]["interrupted"], 1)
            self.assertEqual(report["summary"]["not_run"], 1)
            self.assertTrue((Path(report["run_dir"]) / "report.json").exists())

    def test_launcher_explains_missing_dependencies_without_opening_a_dialog(self):
        with patch("run_paddle_ocr_trial.find_spec", return_value=None), redirect_stdout(StringIO()) as output:
            self.assertEqual(main([]), 2)
        self.assertIn("requirements-ocr.txt", output.getvalue())

    def test_help_prints_vietnamese_even_with_a_legacy_windows_encoding(self):
        script = Path(__file__).resolve().parents[1] / "run_paddle_ocr_trial.py"
        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        result = subprocess.run([sys.executable, "-B", str(script), "--help"],
                                capture_output=True, env=env, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("PP-OCRv6 local", result.stdout.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
