"""Exercise editor launch without opening windows or processing Lab inputs."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from run_image_pipeline import main


class ImageLauncherTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root_dir = Path(self.stack.enter_context(TemporaryDirectory())).resolve()
        self.stack.enter_context(patch('run_image_pipeline.APP_ROOT', self.root_dir))
        self.stack.enter_context(patch('sys.argv', ['run_image_pipeline.py', '--normalize-only']))
        self.stack.enter_context(redirect_stdout(StringIO()))
        self.stack.enter_context(redirect_stderr(StringIO()))
        self.window = self.stack.enter_context(patch('tkinter.Tk')).return_value
        self.mode = self.stack.enter_context(patch('run_image_pipeline._choose_mode', return_value='single'))
        self.batch = self.stack.enter_context(patch('run_image_pipeline.run_all'))
        self.choose = self.stack.enter_context(patch('tkinter.filedialog.askopenfilename'))
        self.ask = self.stack.enter_context(patch('tkinter.simpledialog.askstring'))
        self.info = self.stack.enter_context(patch('tkinter.messagebox.showinfo'))
        self.error = self.stack.enter_context(patch('tkinter.messagebox.showerror'))
        self.warning = self.stack.enter_context(patch('tkinter.messagebox.showwarning'))
        self.normalize = self.stack.enter_context(patch('run_image_pipeline._normalize_selected_input'))

    def select_source(self, relative='data/raw/book_test/07804.jpg'):
        self.source = self.root_dir / relative
        self.choose.return_value = str(self.source)

    def test_cancel_file_does_not_process(self):
        self.choose.return_value = ''
        self.assertEqual(main(), 0)
        self.normalize.assert_not_called()
        self.ask.assert_not_called()
        self.window.destroy.assert_called_once()

    def test_cancel_mode_does_not_process_or_open_picker(self):
        self.mode.return_value = None
        self.assertEqual(main(), 0)
        self.choose.assert_not_called()
        self.normalize.assert_not_called()
        self.batch.assert_not_called()

    def test_all_mode_dispatches_without_single_image_dialogs(self):
        self.mode.return_value = 'all'
        self.batch.return_value = {
            'status': 'completed', 'book_errors': [], 'excluded': [], 'report_warnings': [],
            'report_dir': str(self.root_dir / 'reports'),
            'summary': {'total': 3, 'success': 2, 'skipped_existing': 1,
                        'failed': 0, 'interrupted': 0, 'not_run': 0},
        }
        self.assertEqual(main(), 0)
        self.batch.assert_called_once()
        self.assertEqual(self.batch.call_args.args, (self.root_dir,))
        self.choose.assert_not_called()
        self.ask.assert_not_called()
        self.normalize.assert_not_called()
        self.info.assert_called_once()

    def test_cancel_book_or_stele_does_not_process(self):
        self.select_source()
        for answers in ([None], ['book_test', None]):
            with self.subTest(answers=answers):
                self.ask.side_effect = answers
                self.assertEqual(main(), 0)
        self.normalize.assert_not_called()

    def test_run_suggests_ids_and_passes_user_choices_to_single_file_pipeline(self):
        self.select_source()
        self.ask.side_effect = [' book_test ', ' bia_7804 ']
        self.normalize.return_value = {
            'book_id': 'book_test', 'stele_id': 'bia_7804', 'source_type': 'jpeg',
            'image_id': 'bia_7804.001', 'image_number': '001', 'page_number': None,
            'load_method': 'direct_image', 'source_path': 'synthetic.jpg',
            'image_size_before_resize': [2000, 1000],
            'image_size_after_resize': [2000, 1000],
            'output_jpg': 'data/image_pipeline/book_test/bia_7804/01_prepare/bia_7804.001.jpg',
            'metadata_path': 'synthetic.json', 'source_unchanged': True,
            'preview_path': 'synthetic.preview.html',
        }
        self.assertEqual(main(), 0)
        self.assertEqual([c.kwargs['initialvalue'] for c in self.ask.call_args_list],
                         ['book_test', 'bia_7804'])
        self.normalize.assert_called_once_with(
            self.source, 'bia_7804', book_id='book_test', preview=True,
        )
        self.info.assert_called_once()
        self.error.assert_not_called()

    def test_full_book_at_raw_book_root_is_not_processed(self):
        self.select_source('data/raw/book_test/reference_book.pdf')
        self.assertEqual(main(), 1)
        self.normalize.assert_not_called()
        self.ask.assert_not_called()
        self.error.assert_called_once()

    def test_pending_review_file_is_blocked(self):
        self.select_source('data/raw/book_test/_can_doi_chieu/bia_test/synthetic.jpg')
        self.assertEqual(main(), 1)
        self.normalize.assert_not_called()
        self.ask.assert_not_called()
        self.error.assert_called_once()

    def test_existing_output_error_is_reported_without_retry(self):
        self.select_source()
        self.ask.side_effect = ['book_test', 'bia_7804']
        self.normalize.side_effect = FileExistsError('Output already exists')
        self.assertEqual(main(), 1)
        self.normalize.assert_called_once()
        self.error.assert_called_once()
        self.info.assert_not_called()
        self.window.destroy.assert_called_once()

    def postprocess_report(self):
        return {'stage': 'box_filter_sort', 'status': 'completed', 'book_errors': [],
                'excluded': [], 'report_warnings': [], 'files': [],
                'report_dir': str(self.root_dir / 'reports'),
                'summary': {'total': 0, 'success': 0, 'skipped_existing': 0,
                            'failed': 0, 'interrupted': 0, 'not_run': 0}}

    def test_editor_can_select_local_postprocess_all_without_token_or_ocr(self):
        self.mode.side_effect = ['postprocess', 'all']
        with patch('sys.argv', ['run_image_pipeline.py']), \
             patch('run_image_pipeline.run_filter_sort', return_value=self.postprocess_report()) as run, \
             patch('run_image_pipeline.run_school_pipeline') as school, \
             patch('run_image_pipeline._school_token') as token:
            self.assertEqual(main(), 0)
        run.assert_called_once()
        token.assert_not_called()
        school.assert_not_called()
        self.normalize.assert_not_called()

    def test_postprocess_single_passes_original_box_numbers(self):
        self.mode.return_value = 'single_rules'
        self.select_source()
        self.ask.side_effect = ['book_test', 'bia_7804', '1,3-4', '0']
        with patch('sys.argv', ['run_image_pipeline.py', '--postprocess-only']), \
             patch('run_image_pipeline.run_filter_sort', return_value=self.postprocess_report()) as run, \
             patch('run_image_pipeline.run_school_pipeline') as school:
            self.assertEqual(main(), 0)
        self.assertEqual(run.call_args.kwargs['rules_override'],
                         {'header_present': True, 'header_boxes': [1, 3, 4],
                          'exclude_boxes': [], 'annotation_checked': True})
        school.assert_not_called()

    def test_postprocess_cli_all_is_local_without_gui(self):
        with patch('sys.argv', ['run_image_pipeline.py', '--postprocess-only', '--all']), \
             patch('run_image_pipeline.run_filter_sort', return_value=self.postprocess_report()) as run, \
             patch('run_image_pipeline.run_school_pipeline') as school:
            self.assertEqual(main(), 0)
        self.assertEqual(run.call_args.kwargs['rules_override'], {})
        school.assert_not_called()
        self.mode.assert_not_called()

    def test_postprocess_rejects_indices_for_all_and_conflicting_modes(self):
        for args in (['--postprocess-only', '--all', '--header-boxes', '1'],
                     ['--postprocess-only', '--normalize-only'],
                     ['--header-boxes', '1'],
                     ['--postprocess-only', '--all', '--recheck-completed']):
            with self.subTest(args=args), \
                 patch('sys.argv', ['run_image_pipeline.py', *args]), \
                 self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 2)

    def test_cancel_header_question_does_not_create_results(self):
        self.mode.return_value = 'single_rules'
        self.select_source()
        self.ask.side_effect = ['book_test', 'bia_7804', None]
        with patch('sys.argv', ['run_image_pipeline.py', '--postprocess-only']), \
             patch('run_image_pipeline.run_filter_sort') as run:
            self.assertEqual(main(), 0)
        run.assert_not_called()

    def test_postprocess_single_default_does_not_ask_for_box_numbers(self):
        self.select_source()
        self.ask.side_effect = ['book_test', 'bia_7804']
        with patch('sys.argv', ['run_image_pipeline.py', '--postprocess-only']), \
             patch('run_image_pipeline.run_filter_sort', return_value=self.postprocess_report()) as run:
            self.assertEqual(main(), 0)
        self.assertEqual(self.ask.call_count, 2)
        self.assertEqual(run.call_args.kwargs['rules_override'], {})

    def test_default_run_enables_horizontal_header_pass(self):
        self.mode.return_value = 'all'
        report = self.postprocess_report()
        report['stage'] = 'image_pipeline_through_boxes'
        with patch('sys.argv', ['run_image_pipeline.py']), \
             patch('run_image_pipeline.run_school_pipeline', return_value=report) as run:
            self.assertEqual(main(), 0)
        self.assertTrue(run.call_args.kwargs['include_header'])
        self.assertTrue(run.call_args.kwargs['include_postprocess'])


if __name__ == '__main__':
    unittest.main()
