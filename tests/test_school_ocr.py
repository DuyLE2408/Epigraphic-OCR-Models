"""Synthetic images and mocked HTTP only: no credentials, Lab data or network."""

from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, Mock, patch
import urllib.error

from PIL import Image

from src.image_batch import run_all
from src.image_ocr import normalize_single_input
from src.page_reference import SUPPORTED_BOOKS
from src.school_ocr import (
    BASE_URL, USER_AGENT, OCR_PARAMETERS, RequestGate, SchoolAPIError,
    SchoolClient, SchoolOCR, _parse_boxes, _safe_error, _single_run, run_school_pipeline,
)


BOOK = "van_bia_thoi_tran"
FAKE_TOKEN = "synthetic-test-token-never-send"


def response_data():
    return {"is_success": True, "code": "000000", "message": None,
            "data": {"result_bbox": [
                [[[2, 2], [12, 2], [12, 35], [2, 35]], ["synthetic text", 0.8]]
            ], "result_ocr_text": ["synthetic text"], "result_file_name": "temp-result.jpg"}}


def sources(root, count=1):
    for book in SUPPORTED_BOOKS:
        (root / "data/raw" / book).mkdir(parents=True)
    result = []
    for number in range(1, count + 1):
        path = root / "data/raw" / BOOK / f"{number:05d}.jpg"
        Image.new("RGB", (20, 40), "white").save(path)
        result.append(path)
    return result


class GateTests(unittest.TestCase):
    def test_hourly_limit_persists_across_runs_and_has_resume_time(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            now = 1000000.0
            history = folder / "request_history.json"
            history.write_text(json.dumps({"requests": [now - 1000 + i * 10 for i in range(50)]}))
            before = history.read_bytes()
            gate = RequestGate(folder, clock=lambda: now, sleep=Mock())
            for _ in range(2):
                with self.assertRaises(SchoolAPIError) as stopped:
                    gate.reserve()
                self.assertTrue(stopped.exception.stop_pipeline)
                self.assertIsNotNone(stopped.exception.resume_at)
            self.assertEqual(history.read_bytes(), before)
            self.assertEqual(gate.calls, 0)

    def test_normal_delay_and_reservation_are_saved_without_token(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "request_history.json").write_text(json.dumps({"requests": [99]}))
            sleep = Mock()
            gate = RequestGate(folder, clock=lambda: 100, sleep=sleep)
            gate.reserve()
            sleep.assert_called_once()
            self.assertAlmostEqual(sleep.call_args.args[0], 2.1)
            data = json.loads(gate.path.read_text())
            self.assertEqual(data["requests"], [99, 100])
            self.assertEqual(gate.calls, 1)
            self.assertNotIn("token", gate.path.read_text())

    def test_corrupt_history_stops_before_sending(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "request_history.json").write_text('{"requests":["bad"]}')
            with self.assertRaises(SchoolAPIError):
                RequestGate(folder).reserve()
            (folder / "request_history.json").write_text("broken JSON")
            with self.assertRaises(SchoolAPIError) as stopped:
                RequestGate(folder).reserve()
            self.assertTrue(stopped.exception.stop_pipeline)

    def test_two_process_locks_cannot_overlap_and_lock_releases(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            with _single_run(folder):
                with self.assertRaises(SchoolAPIError):
                    with _single_run(folder):
                        self.fail("Concurrent lock was accepted")
            with _single_run(folder):
                pass


class ClientTests(unittest.TestCase):
    def client(self, payload=None):
        opener = MagicMock()
        response = MagicMock()
        response.status = 200
        response.read.return_value = json.dumps(payload or response_data()).encode()
        opener.open.return_value.__enter__.return_value = response
        gate = Mock()
        client = SchoolClient(lambda: FAKE_TOKEN, gate, opener=opener)
        return client, opener, gate

    def test_ocr_sends_required_headers_and_han_epitaph_parameters(self):
        client, opener, gate = self.client()
        client.ocr("temp.jpg", OCR_PARAMETERS)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, BASE_URL + "/api/web/clc-sinonom/image-ocr")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + FAKE_TOKEN)
        self.assertEqual(request.get_header("User-agent"), USER_AGENT)
        self.assertEqual(json.loads(request.data), {**OCR_PARAMETERS, "file_name": "temp.jpg"})
        gate.reserve.assert_called_once()

    def test_upload_uses_documented_multipart_field_and_preserves_image(self):
        payload = {"is_success": True, "code": "000000", "data": {"file_name": "temp.jpg"}}
        client, opener, _ = self.client(payload)
        with TemporaryDirectory() as temp:
            image = Path(temp) / "bia_1.001.jpg"
            Image.new("RGB", (20, 40)).save(image)
            before = image.read_bytes()
            self.assertEqual(client.upload(image), "temp.jpg")
            self.assertIn(b'name="image_file"', opener.open.call_args.args[0].data)
            self.assertIn(before, opener.open.call_args.args[0].data)
            self.assertEqual(image.read_bytes(), before)

    def test_401_stops_without_echoing_token_or_retrying(self):
        client, opener, _ = self.client()
        opener.open.side_effect = urllib.error.HTTPError(
            BASE_URL, 401, FAKE_TOKEN, {}, BytesIO(FAKE_TOKEN.encode()))
        with self.assertRaises(SchoolAPIError) as error:
            client.ocr("temp.jpg", OCR_PARAMETERS)
        self.assertTrue(error.exception.stop_pipeline)
        self.assertNotIn(FAKE_TOKEN, str(error.exception))
        self.assertEqual(opener.open.call_count, 1)

    def test_429_records_server_cooldown_and_stops(self):
        client, opener, gate = self.client()
        opener.open.side_effect = urllib.error.HTTPError(
            BASE_URL, 429, "limited", {"Retry-After": "120"}, None)
        gate.cooldown.return_value = "2030-01-01T00:00:00+00:00"
        with self.assertRaises(SchoolAPIError) as error:
            client.ocr("temp.jpg", OCR_PARAMETERS)
        gate.cooldown.assert_called_once_with("120")
        self.assertEqual(error.exception.resume_at, gate.cooldown.return_value)

    def test_413_stops_without_retry_or_resizing_the_source(self):
        client, opener, _ = self.client()
        opener.open.side_effect = urllib.error.HTTPError(BASE_URL, 413, 'too large', {}, None)
        with TemporaryDirectory() as temp:
            image = Path(temp) / 'native.jpg'
            Image.new('RGB', (1500, 2300), 'white').save(image)
            before = image.read_bytes()
            with self.assertRaises(SchoolAPIError) as error:
                client.upload(image)
            self.assertTrue(error.exception.stop_pipeline)
            self.assertIn('413', str(error.exception))
            self.assertEqual(opener.open.call_count, 1)
            self.assertEqual(image.read_bytes(), before)

    def test_timeout_stops_without_automatic_retry(self):
        client, opener, _ = self.client()
        opener.open.side_effect = TimeoutError()
        with self.assertRaises(SchoolAPIError) as error:
            client.ocr("temp.jpg", OCR_PARAMETERS)
        self.assertTrue(error.exception.stop_pipeline)
        self.assertEqual(opener.open.call_count, 1)

    def test_http_200_with_failure_envelope_is_not_success(self):
        client, _, _ = self.client({"is_success": False, "code": "bad", "data": None})
        with self.assertRaises(SchoolAPIError):
            client.ocr("temp.jpg", OCR_PARAMETERS)

    def test_errors_keep_diagnostics_but_hide_credentials_and_debug_payload(self):
        message = _safe_error({
            "message": {"title": "Account issue",
                        "text": "Bearer " + FAKE_TOKEN + " person@example.test",
                        "debug_msg": "private debug data"}}, FAKE_TOKEN)
        self.assertIn("Account issue", message)
        self.assertNotIn(FAKE_TOKEN, message)
        self.assertNotIn("person@example.test", message)
        self.assertNotIn("private debug data", message)


class ArtifactTests(unittest.TestCase):
    def prepare(self, root):
        source = sources(root)[0]
        metadata = normalize_single_input(source, "bia_1", book_id=BOOK, app_root=root)
        client = Mock()
        client.upload.return_value = "temp.jpg"
        client.ocr.return_value = response_data()
        return source, metadata, client, SchoolOCR(root, client, progress=lambda _: None)

    def test_outputs_have_four_corners_and_preserve_raw_ocr_without_alignment(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, metadata, client, processor = self.prepare(root)
            original = source.read_bytes()
            result = processor(metadata)
            document = json.loads((root / result["ocr_json"]).read_text())
            box = document["boxes"][0]
            self.assertEqual(box["sino_nom_ocr"], "synthetic text")
            self.assertEqual(len(box["bbox"]), 4)
            self.assertIsNone(box["reading_order"])
            self.assertIsNone(box["patch_path"])
            self.assertIsNone(box["sino_nom"])
            self.assertTrue(document["review_required"])
            self.assertIn("/02_ocr/", result["ocr_json"])
            self.assertNotIn("member3_json", result)
            self.assertEqual(source.read_bytes(), original)
            with Image.open(root / result["bbox_preview"]) as preview:
                self.assertEqual(preview.size, (20, 40))
                self.assertEqual(preview.format, "PNG")
            self.assertEqual(processor(metadata)["ocr_status"], "skipped_existing")
            client.upload.assert_called_once()
            client.ocr.assert_called_once()

    def test_interrupted_ocr_resumes_uploaded_image_without_upload_again(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client, processor = self.prepare(root)
            client.ocr.side_effect = SchoolAPIError("quota", stop_pipeline=True, resume_at="later")
            with self.assertRaises(SchoolAPIError):
                processor(metadata)
            client.ocr.side_effect = None
            self.assertEqual(processor(metadata)["ocr_status"], "success")
            client.upload.assert_called_once()
            self.assertEqual(client.ocr.call_count, 2)

    def test_raw_response_survives_parse_error_and_rerun_spends_no_extra_call(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client, processor = self.prepare(root)
            bad = response_data()
            bad["data"]["result_bbox"][0][0] = [[1, 2]]
            client.ocr.return_value = bad
            for _ in range(2):
                with self.assertRaises(SchoolAPIError):
                    processor(metadata)
            client.upload.assert_called_once()
            client.ocr.assert_called_once()

    def test_changed_image_or_corrupt_cache_is_not_overwritten_or_resent(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client, processor = self.prepare(root)
            result = processor(metadata)
            raw = root / result["ocr_raw_json"]
            raw.write_bytes(b"corrupt")
            with self.assertRaises(SchoolAPIError):
                processor(metadata)
            self.assertEqual(raw.read_bytes(), b"corrupt")
            client.ocr.assert_called_once()
            Image.new("RGB", (20, 40), "red").save(root / metadata["output_jpg"])
            with self.assertRaises(SchoolAPIError):
                processor(metadata)
            client.upload.assert_called_once()

    def test_completed_cache_never_prompts_for_credentials_or_uses_http(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, metadata, _, processor = self.prepare(root)
            processor(metadata)
            token = Mock(side_effect=AssertionError("Token must not be requested"))
            with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("No HTTP")):
                report = run_school_pipeline(root, token_provider=token, selected_source=source,
                                             book_id=BOOK, stele_id="bia_1", progress=lambda _: None)
            self.assertEqual(report["summary"]["skipped_existing"], 1)
            token.assert_not_called()

    def test_box_text_stays_attached_to_polygon_and_bad_counts_are_rejected(self):
        result = response_data()
        result["data"]["result_ocr_text"] = ["different text"]
        self.assertEqual(_parse_boxes(result, "bia_1.001")[0]["sino_nom_ocr"], "synthetic text")
        result["data"]["result_ocr_text"] = []
        with self.assertRaises(SchoolAPIError):
            _parse_boxes(result, "bia_1.001")

    def test_same_stele_id_in_two_books_has_separate_processing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            sources(root)
            other_book = "di_san_han_nom_huyen_dai_loc"
            other = root / "data/raw" / other_book / "00001.jpg"
            Image.new("RGB", (20, 40), "red").save(other)
            client = Mock()
            client.upload.return_value = "temp.jpg"
            client.ocr.return_value = response_data()
            outputs = []
            for book in (BOOK, other_book):
                metadata = normalize_single_input(root / "data/raw" / book / "00001.jpg",
                                                   "bia_1", book_id=book, app_root=root)
                outputs.append(SchoolOCR(root, client, progress=lambda _: None)(metadata))
            self.assertNotEqual(outputs[0]["ocr_json"], outputs[1]["ocr_json"])
            self.assertNotEqual(outputs[0]["bbox_preview"], outputs[1]["bbox_preview"])


class WorkflowTests(unittest.TestCase):
    def test_old_limited_output_is_rejected_before_api_and_preserved(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = sources(root)[0]
            image = root / "data/dataset" / BOOK / "images/bia_1.001.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (20, 40)).save(image)
            record = root / "data/image_pipeline" / BOOK / "bia_1/bia_1.001.json"
            record.parent.mkdir(parents=True)
            metadata = {
                "book_id": BOOK, "stele_id": "bia_1",
                "source_path": source.relative_to(root).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output_jpg": image.relative_to(root).as_posix(),
                "output_jpg_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "metadata_path": record.relative_to(root).as_posix(),
                "source_type": "jpeg", "load_method": "direct_image",
                "image_size_after_resize": [20, 40], "processing": {"max_dim": 1024},
                "dataset_page_number": 1, "page_number": 1,
            }
            record.write_text(json.dumps(metadata), encoding="utf-8")
            originals = {p: p.read_bytes() for p in (source, image)}
            old_metadata = record.read_bytes()
            client = Mock()
            client.upload.return_value = "temp.jpg"
            client.ocr.return_value = response_data()
            processor = SchoolOCR(root, client, progress=lambda _: None)
            first = run_all(root, ocr_processor=processor, progress=lambda _: None)
            self.assertEqual(first["summary"]["failed"], 1)
            self.assertEqual(first["summary"]["success"], 0)
            self.assertEqual(originals, {p: p.read_bytes() for p in originals})
            self.assertEqual(record.read_bytes(), old_metadata)
            self.assertFalse((record.parent / '01_prepare').exists())
            client.upload.assert_not_called()
            client.ocr.assert_not_called()

    def test_quota_stops_all_saves_status_and_next_run_finishes_without_redo(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source_files = sources(root, 3)
            originals = {p: p.read_bytes() for p in source_files}
            client = Mock()
            client.upload.return_value = "temp.jpg"
            client.ocr.side_effect = [
                response_data(), SchoolAPIError("quota", stop_pipeline=True, resume_at="later")]
            processor = SchoolOCR(root, client, progress=lambda _: None)
            report = run_all(root, ocr_processor=processor, progress=lambda _: None)
            self.assertEqual(report["status"], "paused")
            self.assertEqual(report["summary"]["success"], 1)
            self.assertEqual(report["summary"]["not_run"], 2)
            self.assertEqual(report["resume_at_utc"], "later")
            self.assertTrue((Path(report["report_dir"]) / "summary.md").is_file())
            client.reset_mock()
            client.ocr.side_effect = None
            client.ocr.return_value = response_data()
            continued = run_all(root, ocr_processor=processor, progress=lambda _: None)
            self.assertEqual(continued["status"], "completed")
            self.assertEqual(continued["summary"]["success"], 2)
            self.assertEqual(continued["summary"]["skipped_existing"], 1)
            self.assertEqual(client.upload.call_count, 1)
            self.assertEqual(client.ocr.call_count, 2)
            self.assertEqual(originals, {p: p.read_bytes() for p in source_files})

    def test_bad_image_does_not_stop_next_image(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source_files = sources(root, 2)
            source_files[0].write_bytes(b"bad image")
            client = Mock()
            client.upload.return_value = "temp.jpg"
            client.ocr.return_value = response_data()
            report = run_all(root, ocr_processor=SchoolOCR(root, client, progress=lambda _: None),
                             progress=lambda _: None)
            self.assertEqual(report["summary"]["failed"], 1)
            self.assertEqual(report["summary"]["success"], 1)
            client.ocr.assert_called_once()

    def test_default_run_button_selects_full_pipeline_without_normalize_only(self):
        import run_image_pipeline as launcher
        fake = {"status": "completed", "book_errors": [], "excluded": [],
                "report_warnings": [], "report_dir": "synthetic",
                "summary": {"total": 1, "success": 1, "skipped_existing": 0,
                            "failed": 0, "interrupted": 0, "not_run": 0}}
        with patch("sys.argv", ["run_image_pipeline.py"]), patch("tkinter.Tk"), \
                patch("run_image_pipeline._choose_mode", return_value="all"), \
                patch("run_image_pipeline.run_school_pipeline", return_value=fake) as pipeline, \
                patch("run_image_pipeline.run_all") as normalization, \
                patch("tkinter.messagebox.showinfo"), redirect_stdout(StringIO()):
            self.assertEqual(launcher.main(), 0)
        pipeline.assert_called_once()
        self.assertTrue(pipeline.call_args.kwargs['include_postprocess'])
        normalization.assert_not_called()

    def test_token_dialog_masks_input_and_does_not_persist_value(self):
        from run_image_pipeline import _school_token
        with patch.dict("os.environ", {}, clear=True), \
                patch("tkinter.simpledialog.askstring", return_value=FAKE_TOKEN) as dialog:
            self.assertEqual(_school_token(Mock()), FAKE_TOKEN)
            self.assertEqual(dialog.call_args.kwargs["show"], "*")

    def test_single_run_button_passes_selected_raw_source_to_full_pipeline(self):
        import run_image_pipeline as launcher
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "data/raw" / BOOK / "00001.jpg"
            fake = {"status": "completed", "book_errors": [], "excluded": [],
                    "report_warnings": [], "report_dir": "synthetic",
                    "summary": {"total": 1, "success": 1, "skipped_existing": 0,
                                "failed": 0, "interrupted": 0, "not_run": 0}}
            with patch("sys.argv", ["run_image_pipeline.py"]), patch("tkinter.Tk"), \
                    patch("run_image_pipeline.APP_ROOT", root), \
                    patch("run_image_pipeline._choose_mode", return_value="single"), \
                    patch("tkinter.filedialog.askopenfilename", return_value=str(source)), \
                    patch("tkinter.simpledialog.askstring", side_effect=[BOOK, "bia_1"]), \
                    patch("run_image_pipeline.run_school_pipeline", return_value=fake) as pipeline, \
                    patch("run_image_pipeline._normalize_selected_input") as normalize_only, \
                    patch("tkinter.messagebox.showinfo"), redirect_stdout(StringIO()):
                self.assertEqual(launcher.main(), 0)
            self.assertEqual(pipeline.call_args.kwargs["selected_source"], source)
            self.assertEqual(pipeline.call_args.kwargs["book_id"], BOOK)
            self.assertEqual(pipeline.call_args.kwargs["stele_id"], "bia_1")
            self.assertTrue(pipeline.call_args.kwargs['include_postprocess'])
            normalize_only.assert_not_called()


if __name__ == "__main__":
    unittest.main()
