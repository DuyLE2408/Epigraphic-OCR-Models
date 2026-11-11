"""Synthetic tests for supplied page metadata; no book OCR or Lab inputs."""

import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

from run_image_pipeline import _normalize_selected_input
from src.image_ocr import normalize_single_input
from src.page_reference import supplied_pages, get_provided_page, refresh_image_status, collect_sources, infer_stele_id


def source_image(root, book='van_bia_thoi_tran', name='07804.jpg'):
    path = root / 'data/raw' / book / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (30, 60), 'white').save(path)
    return path


def workbook(path, rows):
    cells = []
    for row, code, page in rows:
        cells.append(f'<row r="{row}"><c r="G{row}"><v>{page}</v></c>'
                     f'<c r="H{row}" t="inlineStr"><is><t>{code}</t></is></c></row>')
    content = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
               '<sheetData>' + ''.join(cells) + '</sheetData></worksheet>')
    with ZipFile(path, 'w') as archive:
        archive.writestr('xl/worksheets/sheet1.xml', content)


class SuppliedPageTests(unittest.TestCase):
    def test_compound_filename_uses_primary_code_and_flat_collection_ignores_book(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            names = [f'{code}&04997&04998&04999&05000-8{n}.jpg'
                     for n, code in enumerate(('04997', '04998', '04999', '05000'), 1)]
            for name in names:
                source_image(root, name=name)
            folder = root / 'data/raw/van_bia_thoi_tran'
            (folder / 'STT4_full_book.pdf').write_bytes(b'not read')
            records = collect_sources(root, 'van_bia_thoi_tran')
            self.assertEqual([r['stele_id'] for r in records], ['bia_4997', 'bia_4998', 'bia_4999', 'bia_5000'])
            with self.assertRaises(ValueError):
                infer_stele_id(folder / 'STT4_full_book.pdf')

    def test_flattened_source_keeps_existing_output_identity_without_rewriting_metadata(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            book_id = 'di_san_han_nom_huyen_dai_loc'
            source = source_image(root, book_id, '20367.jpg')
            result = normalize_single_input(source, 'bia_20367', book_id=book_id, app_root=root)
            meta = root / result['metadata_path']
            old = json.loads(meta.read_text(encoding='utf-8'))
            old['source_path'] = 'data/raw/di_san_han_nom_huyen_dai_loc/bia_20367/20367.jpg'
            meta.write_text(json.dumps(old), encoding='utf-8')
            snapshot = meta.read_bytes()
            with patch('src.image_ocr.load_source_image') as load:
                with self.assertRaises(FileExistsError):
                    normalize_single_input(source, 'bia_20367', book_id=book_id, app_root=root)
                load.assert_not_called()
            self.assertEqual(meta.read_bytes(), snapshot)
            report = refresh_image_status(root, book_id)
            with report.open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(list(csv.reader(stream))[1][-2], 'Đã chuẩn hóa')

    def test_provided_excel_page_is_kept_for_each_face_without_incrementing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source_image(root)
            for code in ('04997', '04998', '04999', '05000'):
                source_image(root, name=code + '-face.jpg')
            assignment = root / 'supplied.xlsx'
            workbook(assignment, [(4, '7804', 66), (5, '4997-5001', 322)])
            records = supplied_pages(root, 'van_bia_thoi_tran', assignment=assignment)
            self.assertEqual(len(records), 5)
            for record in records:
                self.assertEqual(record['book_page_number'], 66 if record['code'] == '7804' else 322)
                self.assertEqual(record['page_scope'], 'stele_entry')
                self.assertEqual(record['page_number_source'], 'provided_excel')
            self.assertTrue(all(r['user_confirmed_code_correction'] for r in records if r['code'] != '7804'))

    def test_conflicting_supplied_rows_leave_page_empty(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source_image(root)
            assignment = root / 'supplied.xlsx'
            workbook(assignment, [(4, '7804', 66), (5, '7804', 67)])
            records = supplied_pages(root, 'van_bia_thoi_tran', assignment=assignment, progress=lambda _: None)
            self.assertIsNone(records[0]['book_page_number'])

    def test_missing_excel_does_not_block_normalization(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_image(root)
            provided = get_provided_page(root, 'van_bia_thoi_tran', source,
                                         assignment=root / 'missing.xlsx', progress=lambda _: None)
            result = normalize_single_input(source, 'bia_7804', book_id='van_bia_thoi_tran',
                                            provided_page=provided, app_root=root)
            self.assertIsNone(result['book_page_number'])
            self.assertIsNone(result['page_number'])
            self.assertEqual(result['image_id'], 'bia_7804.001')

    def test_run_never_opens_full_books_or_uses_legacy_ocr_references(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            book_id = 'van_khac_han_nom_thang_long_ha_noi'
            source = source_image(root, book_id, '00099.jpg')
            book = source.parent / 'full_book.pdf'
            book.write_bytes(b'Not a PDF: this file must never be opened')
            old_json = root / 'data/image_pipeline' / book_id / 'page_reference.json'
            old_json.parent.mkdir(parents=True)
            old_json.write_bytes(b'Legacy reference: must be ignored and preserved')
            original_reference = old_json.read_bytes()
            with patch('run_image_pipeline.APP_ROOT', root), \
                    patch('pymupdf.open', side_effect=AssertionError('Must not open full books')), \
                    patch('subprocess.Popen', side_effect=AssertionError('Must not start OCR')):
                result = _normalize_selected_input(source, 'bia_99', book_id=book_id)
            self.assertEqual(result['image_id'], 'bia_99.001')
            self.assertIsNone(result['book_page_number'])
            self.assertIsNone(result['page_number_source'])
            self.assertEqual(old_json.read_bytes(), original_reference)
            with (old_json.parent / 'image_status.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[1][3], '')
            self.assertEqual(rows[1][-2], 'Đã chuẩn hóa')
            output = root / result['output_jpg']
            valid_output = output.read_bytes()
            output.write_bytes(b'corrupted')
            refresh_image_status(root, book_id)
            with (old_json.parent / 'image_status.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(list(csv.reader(stream))[1][-2], 'Thiếu/lỗi output')
            output.write_bytes(valid_output)
            source.write_bytes(b'changed source')
            refresh_image_status(root, book_id)
            with (old_json.parent / 'image_status.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(list(csv.reader(stream))[1][-2], 'Ảnh nguồn đã đổi/thiếu - cần xử lý lại')

    def test_inferred_page_payload_is_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_image(root)
            provided = {'source_path': source.relative_to(root).as_posix(), 'stele_id': 'bia_7804',
                        'book_page_number': 99, 'page_number_source': 'book_page_ocr_candidate'}
            with self.assertRaisesRegex(ValueError, 'Only supplied'):
                normalize_single_input(source, 'bia_7804', book_id='van_bia_thoi_tran',
                                       provided_page=provided, app_root=root)
            self.assertFalse((root / 'data/dataset').exists())

    def test_supplied_page_has_no_effect_on_image_filename_and_can_exceed_999(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_image(root)
            result = normalize_single_input(source, 'bia_7804', book_id='van_bia_thoi_tran',
                                            page_number=1500, app_root=root)
            self.assertEqual(result['image_id'], 'bia_7804.001')
            self.assertEqual(result['book_page_number'], 1500)
            self.assertEqual(result['page_number_source'], 'provided_explicit')
            self.assertEqual(result['image_number'], '001')

    def test_invalid_supplied_page_is_rejected_before_outputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for invalid in (0, -1, True, 1.5, '', 'abc', '../1'):
                with self.subTest(page=invalid), self.assertRaisesRegex(ValueError, 'page_number'):
                    normalize_single_input('missing.jpg', 'bia_test', book_id='book_test',
                                           page_number=invalid, app_root=root)
            self.assertFalse(any(root.iterdir()))

    def test_old_ocr_metadata_never_populates_current_page_column(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            book_id = 'di_san_han_nom_huyen_dai_loc'
            source = source_image(root, book_id, '20367.jpg')
            private = root / 'data/image_pipeline' / book_id / 'bia_20367'
            private.mkdir(parents=True)
            image = root / 'data/dataset' / book_id / 'images' / 'old.053.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'old output')
            metadata = {'source_path': source.relative_to(root).as_posix(),
                        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                        'output_jpg': image.relative_to(root).as_posix(),
                        'output_jpg_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                        'book_page_number': 53, 'page_number_source': 'book_page_ocr_candidate'}
            (private / 'old.053.json').write_text(json.dumps(metadata), encoding='utf-8')
            path = refresh_image_status(root, book_id)
            with path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[1][3], '')
            self.assertEqual(rows[1][-2], 'Đã chuẩn hóa')


if __name__ == '__main__':
    unittest.main()
