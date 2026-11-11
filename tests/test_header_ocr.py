"""Header OCR uses synthetic images and mocked school calls only."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.box_postprocess import filter_and_sort, process_ocr_result, run_filter_sort
from src.header_ocr import HEADER_PARAMETERS, HeaderAwareOCR, select_header_boxes
from src.image_ocr import normalize_single_input, _write_new_files
from src.image_paths import ocr_folder
from src.page_reference import SUPPORTED_BOOKS
from src.school_ocr import OCR_PARAMETERS, SchoolAPIError, SchoolOCR, run_school_pipeline


BOOK = 'van_bia_thoi_tran'


def box(index, x, y, width=40, height=800, text='字'):
    return {'image_id': 'bia_1.001', 'patch_id': f'bia_1.001.{index:03d}',
            'ocr_index': index, 'bbox': [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            'sino_nom_ocr': text, 'confidence': 0.9, 'reading_order': None,
            'patch_path': None, 'sino_nom': None}


def boxes():
    body = [box(i, 300 + (i - 1) * 70, 600) for i in range(1, 6)]
    horizontal = [box(1, 350, 430, 80, 70), box(2, 530, 430, 80, 70),
                  box(3, 400, 900, 150, 40), box(4, 900, 30, 100, 80)]
    return body, horizontal


def response(entries):
    return {'is_success': True, 'code': '000000', 'data': {
        'result_bbox': [[b['bbox'], [b['sino_nom_ocr'], b['confidence']]] for b in entries],
        'result_ocr_text': [b['sino_nom_ocr'] for b in entries]}}


def fixture(root):
    for book in SUPPORTED_BOOKS:
        (root / 'data/raw' / book).mkdir(parents=True)
    source = root / 'data/raw' / BOOK / '00001.jpg'
    Image.new('RGB', (1200, 1600), 'white').save(source)
    metadata = normalize_single_input(source, 'bia_1', book_id=BOOK, app_root=root)
    body, horizontal = boxes()
    client = Mock()
    client.upload.return_value = 'synthetic.jpg'
    client.ocr.side_effect = lambda name, parameters: response(horizontal if parameters['reading_direction'] == 2 else body)
    return source, metadata, client


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HeaderSelectionTests(unittest.TestCase):
    def test_only_large_header_boxes_in_band_are_selected_not_body_or_stamp(self):
        body, horizontal = boxes()
        originals = deepcopy((body, horizontal))
        selected, report = select_header_boxes(body, horizontal, [1200, 1600])
        self.assertEqual([b['source_ocr_index'] for b in selected], [1, 2])
        self.assertEqual(report['status'], 'candidates_found')
        self.assertEqual(len(report['rejected']), 2)
        self.assertEqual((body, horizontal), originals)
        self.assertTrue(all(b['region'] == 'header' for b in selected))

    def test_sparse_or_single_column_does_not_assume_a_header_band(self):
        body, horizontal = boxes()
        selected, report = select_header_boxes(body[:2], horizontal, [1200, 1600])
        self.assertEqual(selected, [])
        self.assertEqual(report['status'], 'region_unknown')

    def test_small_body_text_in_search_band_is_not_promoted_to_title(self):
        body, _ = boxes()
        selected, report = select_header_boxes(body, [box(1, 400, 550, 30, 35)], [1200, 1600])
        self.assertEqual(selected, [])
        self.assertEqual(report['status'], 'not_found')

    def test_title_can_cross_estimated_body_top_while_most_of_glyph_remains_above(self):
        body, _ = boxes()
        candidate = box(1, 400, 540, 80, 90)
        selected, report = select_header_boxes(body, [candidate], [1200, 1600])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['bbox'], candidate['bbox'])
        self.assertTrue(report['automatic_boundary_overlap']['center_must_be_above_body_top'])

    def test_body_character_with_center_below_body_top_is_not_a_header(self):
        body, _ = boxes()
        selected, _ = select_header_boxes(body, [box(1, 400, 580, 80, 90)], [1200, 1600])
        self.assertEqual(selected, [])

    def test_configured_regions_keep_strict_containment_without_automatic_tolerance(self):
        body, _ = boxes()
        selected, _ = select_header_boxes(body, [box(1, 400, 540, 80, 90)], [1200, 1600],
                                           [[220, 400, 700, 600]])
        self.assertEqual(selected, [])

    def test_non_han_and_outside_image_candidates_remain_in_rejected_report(self):
        body, _ = boxes()
        horizontal = [box(1, 350, 430, 80, 70, text='123'), box(2, -20, 430, 100, 70)]
        selected, report = select_header_boxes(body, horizontal, [1200, 1600])
        self.assertEqual(selected, [])
        self.assertEqual(len(report['rejected']), 2)

    def test_configured_region_works_even_when_body_detection_is_sparse(self):
        _, horizontal = boxes()
        selected, report = select_header_boxes([], horizontal, [1200, 1600], [[300, 400, 650, 520]])
        self.assertEqual(len(selected), 2)
        self.assertEqual(report['region_source'], 'configured')

    def test_overlapping_duplicate_headers_do_not_get_two_patches(self):
        body, horizontal = boxes()
        duplicate = deepcopy(horizontal[0])
        duplicate['ocr_index'] = 5
        selected, report = select_header_boxes(body, horizontal + [duplicate], [1200, 1600])
        self.assertEqual(len(selected), 2)
        self.assertIn('overlapping_header_candidate', [r['reason'] for r in report['rejected']])

    def test_header_rows_sort_rtl_and_raw_text_is_not_reversed(self):
        body, horizontal = boxes()
        horizontal[0]['sino_nom_ocr'] = '字文'
        headers, _ = select_header_boxes(body, horizontal, [1200, 1600])
        for i, h in enumerate(headers, 6):
            h['ocr_index'] = i
        result = filter_and_sort(body + headers, [1200, 1600], {'auto_repair': False})
        self.assertEqual([b['ocr_index'] for b in result['boxes'][:2]], [7, 6])
        self.assertEqual(result['boxes'][1]['sino_nom_ocr'], '字文')
        self.assertIn('header_text_direction_needs_review', result['boxes'][1]['review_flags'])
        self.assertIn('automatic_header_needs_review', result['warnings'])
        self.assertEqual([b['reading_order'] for b in result['boxes']], list(range(1, 8)))

    def test_explicit_no_title_rule_rejects_automatic_header_candidates(self):
        body, horizontal = boxes()
        headers, _ = select_header_boxes(body, horizontal, [1200, 1600])
        for i, h in enumerate(headers, 6):
            h['ocr_index'] = i
        result = filter_and_sort(body + headers, [1200, 1600], {'auto_repair': False, 'header_present': False})
        self.assertEqual(len(result['boxes']), len(body))
        self.assertEqual([b['reason'] for b in result['excluded']], ['header_candidate_rejected_by_no_header_rule'] * 2)


class HeaderWorkflowTests(unittest.TestCase):
    def test_general_horizontal_replaces_empty_epitaph_attempt_without_modifying_prior_passes(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            body, horizontal = boxes()
            old_parameters = {**HEADER_PARAMETERS, 'epitaph': 1}
            def respond(name, parameters):
                return response(body if parameters['reading_direction'] == 1 else
                                [] if parameters['epitaph'] == 1 else horizontal)
            client.ocr.side_effect = respond
            with patch('src.header_ocr.HEADER_PARAMETERS', old_parameters), \
                 patch('src.header_ocr.HEADER_PASS_FOLDER', 'header'), \
                 patch('src.school_ocr.SchoolClient', return_value=client):
                old = run_school_pipeline(root, token_provider=Mock(), include_header=True,
                                          include_postprocess=True, progress=lambda _: None)
            self.assertEqual(old['files'][0]['header_box_count'], 0)
            folder = ocr_folder(root, BOOK, 'bia_1', metadata['image_id'])
            old_combination = (folder / 'ocr_combined.json').read_bytes()
            protected = [folder / n for n in ('manifest.json', 'upload.json', 'ocr_raw.json',
                                             'ocr.json', 'bbox_preview.png', 'complete.json')]
            protected += [p for p in (folder / 'header').iterdir() if p.is_file()]
            before = {p: digest(p) for p in protected}
            client.reset_mock()
            with patch('src.school_ocr.SchoolClient', return_value=client):
                new = run_school_pipeline(root, token_provider=Mock(), include_header=True,
                                          include_postprocess=True, progress=lambda _: None)
            self.assertEqual(new['files'][0]['header_box_count'], 2)
            self.assertEqual(HEADER_PARAMETERS['epitaph'], 0)
            client.ocr.assert_called_once_with('synthetic.jpg', HEADER_PARAMETERS)
            client.upload.assert_called_once()
            self.assertEqual(before, {p: digest(p) for p in protected})
            self.assertTrue((folder / 'header/general/ocr_raw.json').is_file())
            history = list((folder / '_combined_history').glob('*/ocr_combined.json'))
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0].read_bytes(), old_combination)

    def test_two_full_image_passes_merge_on_one_image_and_cache_is_reused(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, metadata, client = fixture(root)
            original = source.read_bytes()
            workflow = HeaderAwareOCR(root, client, progress=lambda _: None)
            result = workflow(metadata)
            self.assertEqual(result['header_box_count'], 2)
            self.assertEqual(result['box_count'], 7)
            self.assertEqual(client.ocr.call_args_list[0].args[1], OCR_PARAMETERS)
            self.assertEqual(client.ocr.call_args_list[1].args[1], HEADER_PARAMETERS)
            self.assertEqual(client.upload.call_count, 2)
            for call in client.upload.call_args_list:
                self.assertEqual(call.args[0], root / metadata['output_jpg'])
                self.assertEqual(call.args[0].read_bytes(), original)
            combined = json.loads((root / result['ocr_json']).read_text(encoding='utf-8'))
            self.assertEqual(combined['image_path'], metadata['output_jpg'])
            self.assertEqual(combined['image_size'], [1200, 1600])
            self.assertEqual(len({b['ocr_index'] for b in combined['boxes']}), 7)
            client.reset_mock()
            again = workflow(metadata)
            self.assertEqual(again['ocr_status'], 'skipped_existing')
            client.upload.assert_not_called()
            client.ocr.assert_not_called()
            self.assertEqual(source.read_bytes(), original)

    def test_existing_vertical_results_are_unchanged_and_only_header_is_requested(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            body = SchoolOCR(root, client, progress=lambda _: None)(metadata)
            folder = root / Path(body['ocr_json']).parent
            before = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
            client.reset_mock()
            result = HeaderAwareOCR(root, client, progress=lambda _: None)(metadata)
            client.ocr.assert_called_once_with('synthetic.jpg', HEADER_PARAMETERS)
            client.upload.assert_called_once()
            self.assertEqual(result['header_box_count'], 2)
            self.assertEqual(before, {name: digest(folder / name) for name in before})

    def test_quota_during_header_resumes_without_uploading_or_repeating_body(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            body, horizontal = boxes()
            def fail_header(name, parameters):
                if parameters['reading_direction'] == 2:
                    raise SchoolAPIError('quota', stop_pipeline=True, resume_at='later')
                return response(body)
            client.ocr.side_effect = fail_header
            workflow = HeaderAwareOCR(root, client, progress=lambda _: None)
            with self.assertRaises(SchoolAPIError):
                workflow(metadata)
            self.assertFalse((ocr_folder(root, BOOK, 'bia_1', metadata['image_id']) / 'complete_combined.json').exists())
            client.reset_mock()
            client.ocr.side_effect = lambda *args: response(horizontal)
            result = workflow(metadata)
            self.assertEqual(result['header_box_count'], 2)
            client.ocr.assert_called_once_with('synthetic.jpg', HEADER_PARAMETERS)
            client.upload.assert_not_called()

    def test_combined_output_goes_to_stage_three_and_local_mode_uses_same_source(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, metadata, client = fixture(root)
            result = HeaderAwareOCR(root, client, progress=lambda _: None)(metadata)
            ordered = process_ocr_result(root, root / result['ocr_json'])
            saved = json.loads((root / ordered['sorted_json']).read_text(encoding='utf-8'))
            self.assertEqual([b['source_ocr_index'] for b in saved['boxes'][:2]], [2, 1])
            self.assertTrue(all(b['region'] == 'header' for b in saved['boxes'][:2]))
            self.assertTrue(all(b['bbox'] == boxes()[1][1-i]['bbox'] for i, b in enumerate(saved['boxes'][:2])))
            client.reset_mock()
            local = run_filter_sort(root, selected_source=source, book_id=BOOK, stele_id='bia_1', progress=lambda _: None)
            self.assertEqual(local['summary']['skipped_existing'], 1)
            client.ocr.assert_not_called()

    def test_all_upgrades_completed_body_then_skips_completed_combination_without_api(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, client = fixture(root)
            with patch('src.school_ocr.SchoolClient', return_value=client):
                run_school_pipeline(root, token_provider=Mock(), include_postprocess=True, progress=lambda _: None)
            client.reset_mock()
            with patch('src.school_ocr.SchoolClient', return_value=client):
                report = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                             include_header=True, progress=lambda _: None)
            self.assertEqual(report['summary']['success'], 1)
            self.assertEqual(report['files'][0]['header_box_count'], 2)
            client.ocr.assert_called_once_with('synthetic.jpg', HEADER_PARAMETERS)
            client.reset_mock()
            with patch('src.school_ocr.SchoolClient', return_value=client), \
                    patch('PIL.Image.open', side_effect=AssertionError('No reopening completed image')):
                again = run_school_pipeline(root, token_provider=Mock(), include_postprocess=True,
                                            include_header=True, progress=lambda _: None)
            self.assertEqual(again['previously_completed'], 1)
            self.assertEqual(again['summary']['total'], 0)
            client.ocr.assert_not_called()

    def test_changed_region_reselects_from_cache_and_preserves_prior_combination(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            workflow = HeaderAwareOCR(root, client, progress=lambda _: None)
            initial = workflow(metadata)
            folder = (root / initial['ocr_json']).parent
            old = (root / initial['ocr_json']).read_bytes()
            (folder / 'header_regions.json').write_text(json.dumps({'regions': [[300, 400, 450, 520]]}), encoding='utf-8')
            client.reset_mock()
            result = workflow(metadata)
            self.assertEqual(result['header_box_count'], 1)
            client.ocr.assert_not_called()
            histories = list((folder / '_combined_history').glob('*/ocr_combined.json'))
            self.assertEqual(len(histories), 1)
            self.assertEqual(histories[0].read_bytes(), old)

    def test_corrupt_combination_is_preserved_and_not_replaced(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            workflow = HeaderAwareOCR(root, client, progress=lambda _: None)
            result = workflow(metadata)
            path = root / result['ocr_json']
            path.write_bytes(b'edited output')
            client.reset_mock()
            with self.assertRaises(SchoolAPIError):
                workflow(metadata)
            self.assertEqual(path.read_bytes(), b'edited output')
            client.ocr.assert_not_called()

    def test_no_horizontal_detection_is_reported_as_not_found_not_no_title(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            body, _ = boxes()
            client.ocr.side_effect = lambda name, parameters: response(body if parameters['reading_direction'] == 1 else [])
            result = HeaderAwareOCR(root, client, progress=lambda _: None)(metadata)
            self.assertEqual(result['header_status'], 'not_found')
            self.assertEqual(result['box_count'], len(body))
            self.assertTrue(result['header_review_required'])

    def test_invalid_region_stops_before_any_upload(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            folder = ocr_folder(root, BOOK, 'bia_1', metadata['image_id'])
            folder.mkdir(parents=True)
            (folder / 'header_regions.json').write_text(json.dumps({'regions': [[0, 0, 99999, 99999]]}), encoding='utf-8')
            with self.assertRaises(SchoolAPIError):
                HeaderAwareOCR(root, client, progress=lambda _: None)(metadata)
            client.upload.assert_not_called()

    def test_configured_region_before_first_ocr_is_supported_without_cropping(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            folder = ocr_folder(root, BOOK, 'bia_1', metadata['image_id'])
            folder.mkdir(parents=True)
            (folder / 'header_regions.json').write_text(json.dumps({'regions': [[300, 400, 450, 520]]}), encoding='utf-8')
            result = HeaderAwareOCR(root, client, progress=lambda _: None)(metadata)
            self.assertEqual(result['header_box_count'], 1)
            for call in client.upload.call_args_list:
                self.assertEqual(call.args[0], root / metadata['output_jpg'])

    def test_interruption_before_combined_marker_resumes_without_any_api_request(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            def interrupt_marker(payloads):
                if any(path.name == 'complete_combined.json' for path in payloads):
                    raise KeyboardInterrupt()
                _write_new_files(payloads)
            workflow = HeaderAwareOCR(root, client, progress=lambda _: None)
            with patch('src.header_ocr._write_new_files', side_effect=interrupt_marker):
                with self.assertRaises(KeyboardInterrupt):
                    workflow(metadata)
            client.reset_mock()
            result = workflow(metadata)
            self.assertEqual(result['header_box_count'], 2)
            client.upload.assert_not_called()
            client.ocr.assert_not_called()

    def test_title_already_in_vertical_pass_is_not_duplicated_by_horizontal_pass(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            body, horizontal = boxes()
            prior_title = deepcopy(horizontal[0])
            prior_title['ocr_index'] = 6
            body.append(prior_title)
            client.ocr.side_effect = lambda name, parameters: response(horizontal if parameters['reading_direction'] == 2 else body)
            result = HeaderAwareOCR(root, client, progress=lambda _: None)(metadata)
            self.assertEqual(result['box_count'], 7)
            selected = json.loads((root / result['header_selection']).read_text(encoding='utf-8'))
            self.assertEqual(selected['superseded_vertical_indices'], [6])

    def test_local_combined_processing_rejects_modified_original_api_response(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, client = fixture(root)
            result = HeaderAwareOCR(root, client, progress=lambda _: None)(metadata)
            (root / result['header_ocr_raw_json']).write_bytes(b'edited raw')
            client.reset_mock()
            with self.assertRaises(ValueError):
                process_ocr_result(root, root / result['ocr_json'])
            client.ocr.assert_not_called()


if __name__ == '__main__':
    unittest.main()
