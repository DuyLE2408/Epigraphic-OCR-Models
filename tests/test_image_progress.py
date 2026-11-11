"""Persistent queue tests with synthetic images and mocked school calls."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.image_batch import _validated_existing
from src.image_ocr import normalize_single_input
from src.image_progress import PipelineProgress
from src.page_reference import SUPPORTED_BOOKS
from src.school_ocr import OCR_PARAMETERS, SchoolAPIError, SchoolOCR, run_school_pipeline


BOOK = "van_bia_thoi_tran"


def fixture(root, count):
    for book in SUPPORTED_BOOKS:
        (root / "data/raw" / book).mkdir(parents=True)
    sources = []
    for index in range(1, count + 1):
        path = root / "data/raw" / BOOK / f"{index:05d}.jpg"
        Image.new("RGB", (20, 40)).save(path)
        sources.append(path)
    client = Mock()
    client.upload.return_value = "temp.jpg"
    client.ocr.return_value = {
        "is_success": True, "code": "000000", "data": {
            "result_bbox": [[[[1, 1], [12, 1], [12, 30], [1, 30]], ["synthetic", 0.8]]],
            "result_ocr_text": ["synthetic"]}}
    return sources, client


class ProgressTests(unittest.TestCase):
    def test_native_resolution_reaches_upload_and_box_coordinates_are_not_scaled(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 1)
            Image.new('RGB', (1800, 2600), 'white').save(files[0])
            points = [[1200, 1100], [1280, 1100], [1280, 2400], [1200, 2400]]
            client.ocr.return_value['data']['result_bbox'] = [[points, ['synthetic', 0.8]]]
            uploaded = []
            def upload(path):
                with Image.open(path) as image:
                    uploaded.append(image.size)
                return 'temp.jpg'
            client.upload.side_effect = upload
            with patch('src.school_ocr.SchoolClient', return_value=client):
                result = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                             progress=lambda _: None)
            self.assertEqual(result['summary']['success'], 1)
            self.assertEqual(uploaded, [(1800, 2600)])
            row = result['files'][0]
            document = json.loads((root / row['ocr_json']).read_text(encoding='utf-8'))
            self.assertEqual(document['image_size'], [1800, 2600])
            self.assertEqual(document['boxes'][0]['bbox'], points)
            sorted_doc = json.loads((root / row['sorted_json']).read_text(encoding='utf-8'))
            self.assertEqual(sorted_doc['image_size'], [1800, 2600])
            self.assertEqual(sorted_doc['boxes'][0]['bbox'], points)
            client.upload.assert_called_once()
            client.ocr.assert_called_once()

    def test_missing_preparation_folder_rebuilds_locally_and_reuses_matching_ocr(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 1)
            with patch('src.school_ocr.SchoolClient', return_value=client):
                result = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                             progress=lambda _: None)
            row = result['files'][0]
            stage = (root / row['metadata_path']).parent
            stage.rename(stage.with_name('preparation_backup'))
            client.reset_mock()
            with patch('src.school_ocr.SchoolClient', return_value=client):
                again = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                            progress=lambda _: None)
            self.assertEqual(again['previously_completed'], 0)
            self.assertEqual(again['summary']['success'], 1)
            self.assertTrue((root / row['output_jpg']).is_file())
            client.upload.assert_not_called()
            client.ocr.assert_not_called()

    def test_school_workflow_rejects_pre_ocr_downscale_before_any_request(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 1)
            Image.new('RGB', (1400, 2000), 'white').save(files[0])
            metadata = normalize_single_input(files[0], 'bia_1', book_id=BOOK,
                                              app_root=root, max_dim=1024)
            with self.assertRaisesRegex(SchoolAPIError, 'OCR'):
                SchoolOCR(root, client, progress=lambda _: None)(metadata)
            client.upload.assert_not_called()
            client.ocr.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'max_dim'):
                run_school_pipeline(root, token_provider=Mock(), max_dim=1024)

    def test_selected_cached_ocr_continues_to_stage_three_without_sending_image(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 1)
            metadata = normalize_single_input(files[0], 'bia_1', book_id=BOOK, app_root=root)
            SchoolOCR(root, client, progress=lambda _: None)(metadata)
            client.reset_mock()
            token = Mock(side_effect=AssertionError('No token for cached OCR'))
            logs = []
            with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No HTTP')):
                result = run_school_pipeline(root, token_provider=token, selected_source=files[0],
                                             book_id=BOOK, stele_id='bia_1', include_postprocess=True,
                                             progress=logs.append)
            row = result['files'][0]
            self.assertEqual(row['ocr_status'], 'skipped_existing')
            self.assertEqual(row['box_status'], 'success')
            self.assertEqual(row['status'], 'success')
            self.assertIn('/03_filter_sort/', row['bbox_preview'])
            self.assertTrue((root / row['sorted_json']).is_file())
            self.assertTrue(any('[Bước 3]' in message for message in logs))
            self.assertEqual(result['api_requests_this_run'], 0)
            self.assertEqual(result['box_summary']['completed'], 1)
            token.assert_not_called()
            self.assertEqual(json.loads((root / 'data/image_pipeline/school_api/progress.json').read_text())['target_stage'],
                             '03_filter_sort')

    def test_all_migrates_ocr_only_completed_queue_then_skips_three_stage_results(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 2)
            with patch('src.school_ocr.SchoolClient', return_value=client):
                run_school_pipeline(root, token_provider=Mock(), progress=lambda _: None)
            client.reset_mock()
            with patch('src.school_ocr.SchoolClient', return_value=client):
                result = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                             progress=lambda _: None)
            self.assertEqual(result['summary']['success'], 2)
            self.assertEqual(result['box_summary']['completed'], 2)
            self.assertTrue(all(row['ocr_status'] == 'skipped_existing' for row in result['files']))
            client.upload.assert_not_called()
            client.ocr.assert_not_called()
            with patch('PIL.Image.open', side_effect=AssertionError('Do not reopen completed images')), \
                 patch('src.school_ocr.SchoolOCR.__call__', side_effect=AssertionError('No OCR processing')):
                again = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                            progress=lambda _: None)
            self.assertEqual(again['previously_completed'], 2)
            self.assertEqual(again['summary']['total'], 0)

    def test_failed_stage_three_stays_pending_and_retry_uses_cached_ocr(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 1)
            with patch('src.school_ocr.SchoolClient', return_value=client), \
                 patch('src.box_postprocess.process_ocr_result', side_effect=ValueError('synthetic box failure')):
                failed = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                             progress=lambda _: None)
            self.assertEqual(failed['files'][0]['failure_stage'], 'box_filter_sort')
            state_path = root / 'data/image_pipeline/school_api/progress.json'
            state = json.loads(state_path.read_text())
            self.assertEqual(state['summary'], {'completed': 0, 'pending': 1})
            client.reset_mock()
            with patch('src.school_ocr.SchoolClient', return_value=client):
                result = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                             progress=lambda _: None)
            self.assertEqual(result['files'][0]['box_status'], 'success')
            client.upload.assert_not_called()
            client.ocr.assert_not_called()
            self.assertEqual(json.loads(state_path.read_text())['summary'], {'completed': 1, 'pending': 0})

    def test_full_pipeline_saves_stage_three_before_next_image_hits_quota(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 2)
            good = client.ocr.return_value
            client.ocr.side_effect = [good, SchoolAPIError('quota', stop_pipeline=True, resume_at='later')]
            with patch('src.school_ocr.SchoolClient', return_value=client):
                report = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                             progress=lambda _: None)
            self.assertEqual(report['status'], 'paused')
            self.assertEqual(report['files'][0]['box_status'], 'success')
            self.assertNotIn('box_status', report['files'][1])
            state = json.loads((root / 'data/image_pipeline/school_api/progress.json').read_text())
            self.assertEqual(state['summary'], {'completed': 1, 'pending': 1})

    def test_rollout_finishes_cached_ocr_before_spending_api_on_new_pending_source(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 2)
            with patch('src.school_ocr.SchoolClient', return_value=client):
                run_school_pipeline(root, token_provider=Mock(), selected_source=files[0],
                                    book_id=BOOK, stele_id='bia_1', progress=lambda _: None)
            client.ocr.side_effect = SchoolAPIError('quota', stop_pipeline=True, resume_at='later')
            with patch('src.school_ocr.SchoolClient', return_value=client):
                report = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                             progress=lambda _: None)
            self.assertEqual(report['files'][0]['source_path'], files[0].relative_to(root).as_posix())
            self.assertEqual(report['files'][0]['box_status'], 'success')
            self.assertEqual(report['files'][1]['ocr_status'], 'waiting_quota')

    def test_changed_box_rules_return_image_to_queue_without_repeating_api(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 1)
            with patch('src.school_ocr.SchoolClient', return_value=client):
                report = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                             progress=lambda _: None)
            rules_path = (root / report['files'][0]['sorted_json']).with_name('rules.json')
            rules = json.loads(rules_path.read_text())
            rules['auto_repair'] = False
            rules_path.write_text(json.dumps(rules), encoding='utf-8')
            client.reset_mock()
            with patch('src.school_ocr.SchoolClient', return_value=client):
                again = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                            progress=lambda _: None)
            self.assertEqual(again['summary']['success'], 1)
            self.assertEqual(again['files'][0]['box_status'], 'success')
            client.ocr.assert_not_called()
            client.upload.assert_not_called()

    def test_quota_resume_starts_with_pending_image_not_completed_images(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 3)
            good = client.ocr.return_value
            client.ocr.side_effect = [good, SchoolAPIError("quota", stop_pipeline=True, resume_at="later")]
            with patch("src.school_ocr.SchoolClient", return_value=client):
                first = run_school_pipeline(root, token_provider=Mock(), progress=lambda _: None)
            state_path = root / "data/image_pipeline/school_api/progress.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(first["status"], "paused")
            self.assertEqual(state["summary"], {"completed": 1, "pending": 2})
            self.assertEqual(state["next_source"], files[1].relative_to(root).as_posix())
            client.reset_mock()
            client.ocr.side_effect = None
            with patch("src.school_ocr.SchoolClient", return_value=client), \
                    patch("src.image_batch._validated_existing", wraps=_validated_existing) as validate, \
                    patch("src.image_batch.refresh_image_status", side_effect=AssertionError("No full rescan")):
                second = run_school_pipeline(root, token_provider=Mock(), progress=lambda _: None)
            self.assertEqual(second["previously_completed"], 1)
            self.assertEqual(second["summary"]["total"], 2)
            self.assertEqual(second["summary"]["success"], 2)
            self.assertNotIn(files[0], [call.args[0] for call in validate.call_args_list])
            self.assertEqual(client.upload.call_count, 1)
            self.assertEqual(client.ocr.call_count, 2)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["summary"], {"completed": 3, "pending": 0})
            self.assertIsNone(state["next_source"])

    def test_first_import_and_later_all_done_run_do_not_reopen_finished_images(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 2)
            for index, source in enumerate(files, 1):
                metadata = normalize_single_input(source, f"bia_{index}", book_id=BOOK, app_root=root)
                SchoolOCR(root, client, progress=lambda _: None)(metadata)
            state = PipelineProgress(root, OCR_PARAMETERS)
            self.assertEqual(state.data["summary"], {"completed": 2, "pending": 0})
            token = Mock(side_effect=AssertionError("No token needed"))
            with patch("src.school_ocr.SchoolOCR.__call__", side_effect=AssertionError("No completed work")), \
                    patch("src.image_batch._existing_index", side_effect=AssertionError("No completed metadata")), \
                    patch("PIL.Image.open", side_effect=AssertionError("No image re-open")):
                report = run_school_pipeline(root, token_provider=token, progress=lambda _: None)
            self.assertEqual(report["previously_completed"], 2)
            self.assertEqual(report["summary"]["total"], 0)
            token.assert_not_called()

    def test_changed_input_returns_to_queue_and_does_not_overwrite_output(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files, client = fixture(root, 1)
            with patch("src.school_ocr.SchoolClient", return_value=client):
                first = run_school_pipeline(root, token_provider=Mock(), progress=lambda _: None)
            image = root / first["files"][0]["output_jpg"]
            before = image.read_bytes()
            files[0].write_bytes(b"changed source")
            client.reset_mock()
            with patch("src.school_ocr.SchoolClient", return_value=client):
                report = run_school_pipeline(root, token_provider=Mock(), progress=lambda _: None)
            self.assertEqual(report["summary"]["failed"], 1)
            self.assertEqual(image.read_bytes(), before)
            client.upload.assert_not_called()
            state = PipelineProgress(root, OCR_PARAMETERS)
            self.assertEqual(state.data["summary"], {"completed": 0, "pending": 1})

    def test_corrupt_progress_is_preserved_and_stops_before_api(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 1)
            progress = PipelineProgress(root, OCR_PARAMETERS)
            progress.path.write_text("corrupt", encoding="utf-8")
            with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("No HTTP")):
                with self.assertRaisesRegex(ValueError, "progress.json"):
                    run_school_pipeline(root, token_provider=Mock(), progress=lambda _: None)
            self.assertEqual(progress.path.read_text(encoding="utf-8"), "corrupt")

    def test_explicit_recheck_detects_corrupt_completed_result_without_resending(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, client = fixture(root, 1)
            with patch("src.school_ocr.SchoolClient", return_value=client):
                first = run_school_pipeline(root, token_provider=Mock(), progress=lambda _: None)
            raw = root / first["files"][0]["ocr_raw_json"]
            raw.write_bytes(b"corrupt")
            client.reset_mock()
            with patch("src.school_ocr.SchoolClient", return_value=client):
                report = run_school_pipeline(root, token_provider=Mock(), recheck_completed=True,
                                             progress=lambda _: None)
            self.assertEqual(report["summary"]["failed"], 1)
            client.ocr.assert_not_called()
            self.assertEqual(raw.read_bytes(), b"corrupt")


if __name__ == "__main__":
    unittest.main()
