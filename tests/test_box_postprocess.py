"""Synthetic polygons/images and mocked OCR only; no Lab content or HTTP."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from src.box_postprocess import (
    filter_and_sort, parse_box_indices, process_ocr_result, rules_from_indices, run_filter_sort,
)
from src.image_ocr import normalize_single_input, sort_tran_bia_boxes
from src.image_paths import filter_sort_folder, ocr_folder
from src.school_ocr import SchoolOCR


BOOK = 'van_bia_thoi_tran'


def box(index, x, y, width=10, height=30, text=None, confidence=0.8):
    return {'image_id': 'bia_1.001', 'patch_id': f'bia_1.001.{index:03d}',
            'bbox': [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            'sino_nom_ocr': text if text is not None else str(index), 'confidence': confidence,
            'ocr_index': index, 'reading_order': None, 'sino_nom': None, 'patch_path': None}


def fixture(root, number=1, *, image=None, ocr_boxes=None):
    raw = root / 'data/raw' / BOOK / f'{number:05d}.jpg'
    raw.parent.mkdir(parents=True, exist_ok=True)
    (image if image is not None else Image.new('RGB', (120, 160), 'white')).save(raw)
    stele = f'bia_{number}'
    metadata = normalize_single_input(raw, stele, book_id=BOOK, app_root=root)
    boxes = ocr_boxes if ocr_boxes is not None else [
        box(1, 70, 40), box(2, 30, 40), box(3, 20, 10, 10, 10), box(4, 80, 10, 10, 10)]
    client = Mock()
    client.upload.return_value = 'synthetic.jpg'
    client.ocr.return_value = {'is_success': True, 'code': '000000', 'data': {
        'result_bbox': [[b['bbox'], [b['sino_nom_ocr'], b['confidence']]] for b in boxes],
        'result_ocr_text': [b['sino_nom_ocr'] for b in boxes]}}
    SchoolOCR(root, client, progress=lambda _: None)(metadata)
    return raw, metadata, client, ocr_folder(root, BOOK, stele, metadata['image_id']) / 'ocr.json'


def snapshot(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


class FilterSortGeometryTests(unittest.TestCase):
    def test_confirmed_header_rtl_and_body_columns_ltr_then_top_down(self):
        boxes = [box(1, 70, 45), box(2, 10, 90), box(3, 11, 45),
                 box(4, 25, 5, 8, 8), box(5, 70, 5, 8, 8)]
        original = deepcopy(boxes)
        result = filter_and_sort(boxes, (120, 160), rules_from_indices([4, 5], []))
        self.assertEqual([b['ocr_index'] for b in result['boxes']], [5, 4, 3, 2, 1])
        self.assertEqual([b['reading_order'] for b in result['boxes']], [1, 2, 3, 4, 5])
        self.assertEqual([b['patch_id'] for b in result['boxes']],
                         [f'bia_1.001.{i:03d}' for i in range(1, 6)])
        self.assertEqual(result['order_status'], 'configured')
        for output in result['boxes']:
            source = boxes[output['ocr_index'] - 1]
            for key in ('bbox', 'sino_nom_ocr', 'confidence'):
                self.assertEqual(output[key], source[key])
            self.assertEqual(output['source_patch_id'], source['patch_id'])
        self.assertEqual(boxes, original)

    def test_only_invalid_outside_and_exact_duplicates_are_automatically_removed(self):
        boxes = [box(1, 1, 1, 1, 1, confidence=0.001), box(2, 20, 20),
                 box(3, 20, 20, text='2'), box(4, 130, 20), box(5, 40, 40)]
        boxes[4]['bbox'] = [[40, 40], [50, 70], [40, 70], [50, 40]]
        boxes.append(box(6, 20, 20, text='different OCR'))
        result = filter_and_sort(boxes, (120, 160))
        self.assertEqual({b['ocr_index'] for b in result['boxes']}, {1, 2, 6})
        self.assertEqual([b['reason'] for b in result['excluded']],
                         ['exact_duplicate', 'outside_image', 'invalid_polygon'])
        self.assertEqual(result['order_status'], 'provisional')
        self.assertTrue(result['review_required'])

    def test_partial_outside_is_retained_and_flagged(self):
        result = filter_and_sort([box(1, -3, 10)], (120, 160), rules_from_indices([], []))
        self.assertEqual(len(result['boxes']), 1)
        self.assertIn('partly_outside_image', result['boxes'][0]['review_flags'])
        self.assertEqual(result['order_status'], 'provisional')

    def test_explicit_regions_are_conservative_containment_not_overlap(self):
        boxes = [box(1, 10, 5, 10, 10), box(2, 70, 5, 10, 10),
                 box(3, 20, 40), box(4, 25, 90)]
        rules = {'header_regions': [[0, 0, 100, 20]], 'exclude_regions': [[15, 35, 32, 75]],
                 'annotation_checked': True}
        result = filter_and_sort(boxes, (120, 160), rules)
        self.assertEqual([b['ocr_index'] for b in result['boxes']], [2, 1, 4])
        self.assertEqual(result['excluded'][0]['reason'], 'excluded_annotation_region')

    def test_header_text_is_not_reversed_and_direction_is_flagged(self):
        result = filter_and_sort([box(1, 10, 10, 80, 10, text='ABC')], (120, 160),
                                 rules_from_indices([1], []))
        self.assertEqual(result['boxes'][0]['sino_nom_ocr'], 'ABC')
        self.assertIn('header_text_direction_needs_review', result['warnings'])

    def test_rules_reject_unknown_indices_and_conflicting_header_and_exclusion(self):
        for rules in ({'header_boxes': [99]}, {'header_boxes': [1], 'exclude_boxes': [1]},
                      {'header_present': False, 'header_boxes': [1]},
                      {'header_present': True}, {'exclude_regions': [[1, 1, 0, 5]]}):
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                filter_and_sort([box(1, 10, 10)], (120, 160), rules)

    def test_nonfinite_and_degenerate_polygons_are_excluded(self):
        boxes = [box(1, 0, 0, 0, 20), box(2, 10, 10)]
        boxes[1]['bbox'][0][0] = float('nan')
        result = filter_and_sort(boxes, (120, 160))
        self.assertEqual(len(result['excluded']), 2)
        self.assertIn('no_usable_boxes', result['warnings'])

    def test_header_adapter_keeps_readme_signature_and_defaults_to_header(self):
        original = [box(1, 10, 5, 10, 10), box(2, 60, 5, 10, 10)]
        result = sort_tran_bia_boxes(original)
        self.assertEqual([b['ocr_index'] for b in result], [2, 1])
        self.assertNotIn('region', original[0])

    def test_box_index_input_distinguishes_unknown_from_confirmed_none(self):
        self.assertEqual(parse_box_indices('1,2,5-7'), [1, 2, 5, 6, 7])
        self.assertEqual(parse_box_indices('0'), [])
        self.assertIsNone(parse_box_indices(' '))
        for value in ('-1', '7-2', '0,1', '1000', 'abc'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_box_indices(value)


class FilterSortStorageTests(unittest.TestCase):
    def test_main_preview_excludes_unapplied_proposals_and_legacy_cache_upgrades_safely(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            image = Image.new('RGB', (120, 240), 'white')
            draw = ImageDraw.Draw(image)
            for x in (30, 80):
                for y in range(20, 220, 20):
                    draw.rectangle((x + 4, y, x + 5, y + 12), fill='black')
                    draw.rectangle((x + 13, y, x + 14, y + 12), fill='black')
                    draw.line((x + 4, y + 6, x + 14, y + 6), fill='black')
            _, _, _, path = fixture(root, image=image, ocr_boxes=[box(1, 30, 20, 20, 60), box(2, 80, 20, 20, 200)])
            row = process_ocr_result(root, path)
            self.assertGreater(row['repair_proposals'], 0)
            with Image.open(root / row['bbox_preview']) as preview:
                self.assertNotIn((255, 255, 0), preview.convert('RGB').get_flattened_data())
            with Image.open(root / row['proposals_preview']) as proposals:
                self.assertIn((255, 255, 0), proposals.convert('RGB').get_flattened_data())
            folder = (root / row['sorted_json']).parent
            marker = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
            marker['binding'].update(algorithm_version='2.0', repair_version='1.0')
            marker['sha256'].pop('proposals_preview')
            (folder / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
            (folder / 'proposals_preview.png').unlink()
            before = (folder / 'boxes.json').read_bytes()
            upgraded = process_ocr_result(root, path)
            self.assertEqual(upgraded['status'], 'success')
            self.assertTrue((root / upgraded['proposals_preview']).is_file())
            self.assertEqual(next(folder.glob('_history/*/boxes.json')).read_bytes(), before)
            self.assertEqual(process_ocr_result(root, path)['status'], 'skipped_existing')

    def test_automatic_repair_is_saved_reported_and_cached_without_another_ocr_call(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            image = Image.new('RGB', (120, 160), 'white')
            draw = ImageDraw.Draw(image)
            for x in (30, 80):
                for y in range(20, 140, 20):
                    draw.rectangle((x + 4, y, x + 5, y + 12), fill='black')
                    draw.rectangle((x + 13, y, x + 14, y + 12), fill='black')
                    draw.line((x + 4, y + 6, x + 14, y + 6), fill='black')
            source, metadata, client, path = fixture(root, image=image, ocr_boxes=[
                box(1, 30, 20, 20, 40), box(2, 30, 70, 20, 70), box(3, 80, 20, 20, 120)])
            before = snapshot(root.rglob('*'))
            with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No HTTP')):
                report = run_filter_sort(root, progress=lambda _: None)
                row = report['files'][0]
                self.assertEqual(row['box_count'], 2)
                self.assertGreaterEqual(row['repair_applied'], 1)
                self.assertGreaterEqual(row['ocr_refresh_count'], 1)
                self.assertEqual(run_filter_sort(root, progress=lambda _: None)['summary']['skipped_existing'], 1)
            doc = json.loads((root / row['sorted_json']).read_text(encoding='utf-8'))
            self.assertEqual(doc['boxes'][0]['source_ocr_indices'], [1, 2])
            details = json.loads((root / row['sorted_json']).with_name('report.json').read_text(encoding='utf-8'))
            self.assertTrue(details['coordinates_changed'])
            self.assertTrue(details['ocr_text_changed'])
            self.assertTrue(details['source_ocr_unchanged'])
            self.assertEqual(before, snapshot(Path(p) for p in before))
            client.ocr.assert_called_once()

    def test_source_preservation_cache_and_no_http_or_credentials(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, metadata, client, path = fixture(root)
            before = snapshot(root.rglob('*'))
            with patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('No HTTP')):
                report = run_filter_sort(root, selected_source=source, book_id=BOOK,
                                         stele_id='bia_1', rules_override=rules_from_indices([3, 4], []),
                                         progress=lambda _: None)
                self.assertEqual(report['summary']['success'], 1)
                again = run_filter_sort(root, progress=lambda _: None)
            self.assertEqual(again['summary']['skipped_existing'], 1)
            self.assertEqual(before, snapshot(Path(p) for p in before))
            client.upload.assert_called_once()
            client.ocr.assert_called_once()
            folder = filter_sort_folder(root, BOOK, 'bia_1', metadata['image_id'])
            doc = json.loads((folder / 'boxes.json').read_text(encoding='utf-8'))
            self.assertEqual([b['ocr_index'] for b in doc['boxes']], [4, 3, 2, 1])
            self.assertTrue((folder / 'complete.json').is_file())
            with Image.open(folder / 'bbox_preview.png') as image:
                self.assertEqual(image.size, (120, 160))
            for filename in ('report.json', 'report.csv', 'summary.md'):
                self.assertTrue((Path(report['report_dir']) / filename).is_file())
            self.assertFalse((root / 'data/image_pipeline/school_api').exists())

    def test_changed_rules_archive_previous_output_and_preserve_original_ocr(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, metadata, _, path = fixture(root)
            first = process_ocr_result(root, path)
            folder = root / first['sorted_json']
            first_content, ocr_content = folder.read_bytes(), path.read_bytes()
            process_ocr_result(root, path, rules_override=rules_from_indices([3, 4], [2]))
            history = list(folder.parent.glob('_history/*/boxes.json'))
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0].read_bytes(), first_content)
            self.assertEqual(path.read_bytes(), ocr_content)
            self.assertEqual(process_ocr_result(root, path)['status'], 'skipped_existing')

    def test_modified_output_is_preserved_and_reported_not_overwritten(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, path = fixture(root)
            first = process_ocr_result(root, path)
            output = root / first['sorted_json']
            output.write_bytes(b'manually edited')
            report = run_filter_sort(root, progress=lambda _: None)
            self.assertEqual(report['summary']['failed'], 1)
            self.assertEqual(report['files'][0]['error_type'], 'ValueError')
            self.assertEqual(output.read_bytes(), b'manually edited')

    def test_batch_continues_after_bad_ocr_and_rejects_shared_indices(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, _, path = fixture(root)
            fixture(root, 2)
            path.write_bytes(b'bad OCR')
            report = run_filter_sort(root, progress=lambda _: None)
            self.assertEqual(report['summary']['success'], 1)
            self.assertEqual(report['summary']['failed'], 1)
            with self.assertRaises(ValueError):
                run_filter_sort(root, rules_override=rules_from_indices([1], []))

    def test_interruption_reports_current_and_pending_images(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            fixture(root, 2)
            with patch('src.box_postprocess.process_ocr_result', side_effect=KeyboardInterrupt):
                report = run_filter_sort(root, progress=lambda _: None)
            self.assertEqual(report['status'], 'interrupted')
            self.assertEqual(report['summary']['interrupted'], 1)
            self.assertEqual(report['summary']['not_run'], 1)


if __name__ == '__main__':
    unittest.main()
