"""Batch tests use only temporary synthetic images/PDFs; never real Lab input."""

from contextlib import redirect_stdout, redirect_stderr
import csv
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image
import pymupdf

from run_image_pipeline import main
from src.image_batch import run_all
from src.page_reference import SUPPORTED_BOOKS


def folders(root):
    for book in SUPPORTED_BOOKS:
        (root / 'data/raw' / book).mkdir(parents=True)


def jpeg(root, code, book='di_san_han_nom_huyen_dai_loc'):
    path = root / 'data/raw' / book / f'{code}.jpg'
    Image.new('RGB', (80, 160), 'white').save(path)
    return path


class ImageBatchTests(unittest.TestCase):
    def test_all_continues_after_errors_and_reports_every_file(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folders(root)
            jpeg(root, '00001')
            jpeg(root, '00003', 'van_khac_han_nom_thang_long_ha_noi')
            bad = root / 'data/raw/di_san_han_nom_huyen_dai_loc/00002.jpg'
            bad.write_bytes(b'broken image')
            pdf = root / 'data/raw/van_bia_thoi_tran/00004.pdf'
            with pymupdf.open() as document:
                document.new_page()
                document.new_page()
                document.save(pdf)
            book = root / 'data/raw/van_bia_thoi_tran/STT4_book.pdf'
            book.write_bytes(b'full book must not be opened')
            original = {p: p.read_bytes() for p in (root / 'data/raw').rglob('*') if p.is_file()}
            report = run_all(root, progress=lambda _: None)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['summary'], {'total': 4, 'success': 2, 'skipped_existing': 0,
                                                'failed': 2, 'interrupted': 0, 'not_run': 0})
            self.assertEqual(original, {p: p.read_bytes() for p in original})
            failed = [r for r in report['files'] if r['status'] == 'failed']
            self.assertEqual({Path(r['source_path']).name for r in failed}, {'00002.jpg', '00004.pdf'})
            self.assertTrue(all(r['reason'] and r['error_type'] for r in failed))
            self.assertTrue(any('PDF đầu vào' in r['reason'] for r in failed))
            self.assertTrue(any('STT4_book.pdf' in text for text in report['excluded']))
            directory = Path(report['report_dir'])
            saved = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['summary'], report['summary'])
            with (directory / 'report.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(len(list(csv.reader(stream))), 5)
            summary = (directory / 'summary.md').read_text(encoding='utf-8')
            self.assertIn('00002.jpg', summary)
            self.assertIn('00004.pdf', summary)

    def test_rerun_skips_valid_results_without_changing_them_and_keeps_old_report(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folders(root)
            jpeg(root, '00001')
            jpeg(root, '00002')
            first = run_all(root, progress=lambda _: None)
            outputs = list((root / 'data/image_pipeline').glob('*/bia_*/01_prepare/*.jpg'))
            metadata = list((root / 'data/image_pipeline').glob('*/bia_*/01_prepare/*.json'))
            self.assertEqual(len(outputs), 2)
            self.assertEqual(len(metadata), 2)
            snapshot = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in outputs + metadata}
            old_report = Path(first['report_dir']) / 'report.json'
            original_report = old_report.read_bytes()
            second = run_all(root, progress=lambda _: None)
            self.assertEqual(second['summary']['success'], 0)
            self.assertEqual(second['summary']['skipped_existing'], 2)
            self.assertEqual(second['summary']['failed'], 0)
            self.assertEqual(snapshot, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in snapshot})
            self.assertNotEqual(first['report_dir'], second['report_dir'])
            self.assertEqual(old_report.read_bytes(), original_report)

    def test_changed_source_is_an_error_not_a_skip_or_overwrite(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folders(root)
            source = jpeg(root, '00001')
            first = run_all(root, progress=lambda _: None)
            output = root / first['files'][0]['output_jpg']
            before = output.read_bytes()
            Image.new('RGB', (30, 40), 'red').save(source)
            second = run_all(root, progress=lambda _: None)
            self.assertEqual(second['summary']['failed'], 1)
            self.assertEqual(second['summary']['skipped_existing'], 0)
            self.assertIn('nguồn đã đổi', second['files'][0]['reason'])
            self.assertEqual(output.read_bytes(), before)

    def test_orphan_jpg_without_metadata_is_not_counted_as_complete(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folders(root)
            jpeg(root, '00001')
            output = root / 'data/image_pipeline/di_san_han_nom_huyen_dai_loc/bia_1/01_prepare/bia_1.001.jpg'
            output.parent.mkdir(parents=True)
            output.write_bytes(b'unknown old output')
            report = run_all(root, progress=lambda _: None)
            self.assertEqual(report['summary']['failed'], 1)
            self.assertEqual(report['summary']['success'], 0)
            self.assertIn('thiếu JSON', report['files'][0]['reason'])
            self.assertEqual(output.read_bytes(), b'unknown old output')

    def test_relocated_source_can_reuse_legacy_metadata_but_changed_settings_cannot(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folders(root)
            jpeg(root, '00001')
            first = run_all(root, progress=lambda _: None)
            metadata_path = root / first['files'][0]['metadata_path']
            data = json.loads(metadata_path.read_text(encoding='utf-8'))
            data['source_path'] = 'data/raw/di_san_han_nom_huyen_dai_loc/bia_1/00001.jpg'
            metadata_path.write_text(json.dumps(data), encoding='utf-8')
            second = run_all(root, progress=lambda _: None)
            self.assertEqual(second['summary']['skipped_existing'], 1)
            third = run_all(root, max_dim=512, progress=lambda _: None)
            self.assertEqual(third['summary']['failed'], 1)
            self.assertIn('Cấu hình', third['files'][0]['reason'])

    def test_interrupt_keeps_checkpoint_and_marks_unprocessed_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folders(root)
            jpeg(root, '00001')
            jpeg(root, '00002')
            with patch('src.image_batch.normalize_single_input', side_effect=KeyboardInterrupt):
                report = run_all(root, progress=lambda _: None)
            self.assertEqual(report['status'], 'interrupted')
            self.assertEqual(report['summary']['interrupted'], 1)
            self.assertEqual(report['summary']['not_run'], 1)
            self.assertEqual(report['summary']['success'], 0)
            self.assertTrue((Path(report['report_dir']) / 'report.json').is_file())

    def test_bad_filename_is_reported_and_missing_book_does_not_stop_other_books(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'data/raw/di_san_han_nom_huyen_dai_loc'
            folder.mkdir(parents=True)
            jpeg(root, '00001')
            Image.new('RGB', (20, 40)).save(folder / 'bad_name.jpg')
            report = run_all(root, progress=lambda _: None)
            self.assertEqual(report['summary']['total'], 2)
            self.assertEqual(report['summary']['success'], 1)
            self.assertEqual(report['summary']['failed'], 1)
            self.assertEqual(len(report['book_errors']), 2)
            self.assertTrue(any('filename' in r['reason'] for r in report['files'] if r['status'] == 'failed'))

    def test_cli_all_dispatch_and_mutually_exclusive_input(self):
        fake = {'status': 'completed', 'book_errors': [], 'excluded': [], 'report_warnings': [],
                'report_dir': 'synthetic_reports',
                'summary': {'total': 1, 'success': 1, 'skipped_existing': 0, 'failed': 0,
                            'interrupted': 0, 'not_run': 0}}
        with patch('sys.argv', ['run_image_pipeline.py', '--normalize-only', '--all']), \
                patch('run_image_pipeline.run_all', return_value=fake) as batch, redirect_stdout(StringIO()):
            self.assertEqual(main(), 0)
            batch.assert_called_once()
        with patch('sys.argv', ['run_image_pipeline.py', '--normalize-only', '--all', '--input', '00001.jpg']), \
                patch('run_image_pipeline.run_all') as batch, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                main()
            batch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
