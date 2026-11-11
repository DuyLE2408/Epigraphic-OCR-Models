"""Run raw images through normalization, school OCR and local box previews."""

import argparse
import getpass
import json
import os
from pathlib import Path
import sys

from src.image_ocr import APP_ROOT, normalize_single_input
from src.page_reference import SUPPORTED_BOOKS, infer_stele_id, get_provided_page, refresh_image_status
from src.image_batch import run_all
from src.school_ocr import run_school_pipeline
from src.box_postprocess import parse_box_indices, rules_from_indices, run_filter_sort


def _choose_mode(root, *, allow_postprocess=True, allow_region_rules=False):
    from tkinter import Toplevel, ttk
    dialog = Toplevel(root)
    dialog.title('Chế độ xử lý ảnh')
    dialog.resizable(False, False)
    selection = {'mode': None}

    def choose(mode):
        selection['mode'] = mode
        dialog.destroy()

    frame = ttk.Frame(dialog, padding=20)
    frame.pack()
    ttk.Label(frame, text='Bạn muốn xử lý ảnh nào?').pack(pady=(0, 12))
    one = ttk.Button(frame, text='Chọn 1 ảnh', command=lambda: choose('single'))
    one.pack(fill='x', pady=4)
    ttk.Button(frame, text='All — toàn bộ 3 sách', command=lambda: choose('all')).pack(fill='x', pady=4)
    if allow_postprocess:
        ttk.Button(frame, text='Lọc / tự sửa / sắp box đã OCR (không gọi API)',
                   command=lambda: choose('postprocess')).pack(fill='x', pady=4)
    if allow_region_rules:
        ttk.Button(frame, text='Chọn 1 ảnh + khai báo vùng (tùy chọn)',
                   command=lambda: choose('single_rules')).pack(fill='x', pady=4)
    ttk.Label(frame, text='Giữ kết quả hợp lệ đã có và lưu tiến độ từng ảnh.').pack(pady=(12, 0))
    dialog.protocol('WM_DELETE_WINDOW', lambda: choose(None))
    dialog.grab_set()
    one.focus_set()
    root.wait_window(dialog)
    return selection['mode']


def _batch_message(report):
    count = report['summary']
    message = (f"Tổng số file: {count['total']}\nThành công mới: {count['success']}\n"
            f"Đã có hợp lệ, bỏ qua: {count['skipped_existing']}\nLỗi: {count['failed']}\n"
            f"Bị ngắt: {count['interrupted']}\nChưa chạy: {count['not_run']}\n"
            f"Lỗi thư mục/dữ liệu hỗ trợ: {len(report['book_errors'])}\n\n"
            f"File/thư mục loại trừ: {len(report['excluded'])}; cảnh báo báo cáo: {len(report['report_warnings'])}\n\n"
            f"Báo cáo chi tiết:\n{report['report_dir']}\n\nMở summary.md hoặc report.csv.")
    if report.get('stage') in {'box_filter_sort', 'image_pipeline_through_boxes'}:
        provisional = sum(row.get('order_status') == 'provisional' for row in report['files'])
        repaired = sum(row.get('repair_applied', 0) for row in report['files'])
        proposals = sum(row.get('repair_proposals', 0) for row in report['files'])
        refresh = sum(row.get('ocr_refresh_count', 0) for row in report['files'])
        tail_review = sum(row.get('tail_review_count', 0) for row in report['files'])
        message += (f'\n\nRiêng bước 03 lọc/tự sửa/sắp box chạy cục bộ; không gọi API.\n'
                    f'Tự sửa: {repaired}; đề xuất cần xem: {proposals}; box cần OCR lại: {refresh}.\n'
                    f'Đuôi box cần đối chiếu văn bản: {tail_review}.\n'
                    f'{provisional} ảnh có vùng, thứ tự hoặc box cần kiểm tra.\n'
                    'Mọi kết quả vẫn cần kiểm tra trực quan trước khi crop.')
        if report.get('stage') == 'image_pipeline_through_boxes':
            message += '\nLuồng Run: 01_prepare → 02_ocr → 03_filter_sort.'
    if report.get('stop_reason'):
        message += '\n\nĐã lưu tiến độ: ' + report['stop_reason']
    if report.get('progress_file'):
        message += (f"\n\nĐã hoàn thành từ trước: {report['previously_completed']}"
                    f"\nTiến độ chung: {report['progress_file']}")
    for row in report.get('files', []):
        if 'header_status' in row:
            message += (f"\nTrán bia {row.get('image_id')}: {row['header_box_count']} box đề xuất; "
                        f"{row['header_status']} — cần kiểm tra.")
    if len(report.get('files', [])) == 1 and report['files'][0].get('bbox_preview'):
        message += '\n\nẢnh xem box:\n' + str(APP_ROOT / report['files'][0]['bbox_preview'])
    return message


def _batch_exit_code(report):
    if report['status'] == 'interrupted':
        return 130
    if report['status'] in {'paused', 'stopped'}:
        return 1
    return 1 if report['summary']['failed'] or report['book_errors'] else 0


def _normalize_selected_input(input_path, stele_id, output_path=None, **options):
    """Normalize immediately; supplied pages are optional descriptive metadata."""
    book_id = options['book_id']
    if book_id not in SUPPORTED_BOOKS:
        raise ValueError('Unknown book ID; choose one of the three configured raw book folders.')
    source = Path(input_path)
    source = (APP_ROOT / source).resolve() if not source.is_absolute() else source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Input not found: {source}')
    if source.parent != (APP_ROOT / 'data/raw' / book_id).resolve():
        raise ValueError('Input must be directly inside data/raw/<book-id>/.')
    if stele_id != infer_stele_id(source):
        raise ValueError('Stele ID must match the first code in the source filename.')
    record = get_provided_page(APP_ROOT, book_id, source)
    options.setdefault('app_root', APP_ROOT)
    try:
        return normalize_single_input(input_path, stele_id, output_path,
                                      provided_page=record, **options)
    finally:
        report = refresh_image_status(APP_ROOT, book_id)
        print(f'Image status table: {report}', flush=True)


def _print_summary(metadata: dict) -> None:
    print(json.dumps({key: metadata[key] for key in (
        "book_id", "stele_id", "image_id", "image_number", "page_number", "source_type", "load_method", "source_path",
        "image_size_before_resize", "image_size_after_resize", "output_jpg",
        "metadata_path", "source_unchanged"
    )}, ensure_ascii=False, indent=2))
    if "preview_path" in metadata:
        print("Local preview:", metadata["preview_path"])


def _school_token(root=None):
    """Never write credentials to source, reports, or command history."""
    token = os.environ.get('KIMHANNOM_TOKEN', '').strip()
    if token:
        return token
    if root is None:
        return getpass.getpass('Dan gia tri cookie token Kim Han Nom (nhap an): ')
    from tkinter import simpledialog
    return simpledialog.askstring(
        'Token Kim Hán Nôm',
        'Dán giá trị cookie token sau khi đăng nhập website trường.\n'
        'Token chỉ dùng trong phiên chạy này, không lưu vào file.',
        show='*', parent=root)


def _run_interactive(normalize_only=False, postprocess_only=False) -> int:
    """Choose one local input when launched with the editor's Run button."""
    try:
        from tkinter import Tk, filedialog, messagebox, simpledialog
        root = Tk()
        root.withdraw()
    except Exception as exc:
        print(f"Cannot open the file chooser: {exc}. Use --help for Terminal mode.",
              file=sys.stderr)
        return 1

    try:
        mode = _choose_mode(root, allow_postprocess=not (normalize_only or postprocess_only),
                            allow_region_rules=postprocess_only)
        if mode == 'postprocess':
            postprocess_only = True
            mode = _choose_mode(root, allow_postprocess=False, allow_region_rules=True)
        if mode is None:
            return 0
        if mode == 'all':
            if postprocess_only:
                report = run_filter_sort(APP_ROOT, progress=lambda s: print(s, flush=True))
            elif normalize_only:
                report = run_all(APP_ROOT, preview=True, progress=lambda s: print(s, flush=True))
            else:
                report = run_school_pipeline(APP_ROOT, token_provider=lambda: _school_token(root),
                                             progress=lambda s: print(s, flush=True), include_postprocess=True, include_header=True)
            print(_batch_message(report))
            show = messagebox.showwarning if _batch_exit_code(report) or report['report_warnings'] else messagebox.showinfo
            show('Kết quả pipeline', _batch_message(report), parent=root)
            return _batch_exit_code(report)
        selected = filedialog.askopenfilename(
            parent=root, title="Chọn một PDF hoặc ảnh JPG trong raw",
            initialdir=str(APP_ROOT / "data/raw"),
            filetypes=[("PDF / JPG / JPEG", "*.pdf *.jpg *.jpeg *.PDF *.JPG *.JPEG")],
        )
        if not selected:
            return 0
        source = Path(selected).resolve()
        suggested_book = suggested_stele = ""
        if source.is_relative_to((APP_ROOT / "data/raw").resolve()):
            parts = source.relative_to((APP_ROOT / "data/raw").resolve()).parts
            if "_can_doi_chieu" in parts:
                raise ValueError("File đang ở _can_doi_chieu; cần đối chiếu trước khi xử lý.")
            if len(parts) != 2:
                raise ValueError("Hãy chọn ảnh/PDF nằm trực tiếp trong thư mục sách ở data/raw/.")
            suggested_book, suggested_stele = parts[0], infer_stele_id(source)
        book_id = simpledialog.askstring(
            "Mã sách", "Kiểm tra mã sách (tên thư mục sách trong data/raw):",
            initialvalue=suggested_book, parent=root,
        )
        if book_id is None:
            return 0
        stele_id = simpledialog.askstring(
            "Mã bia", "Kiểm tra mã bia (ví dụ bia_7804):",
            initialvalue=suggested_stele, parent=root,
        )
        if stele_id is None:
            return 0
        if postprocess_only:
            rules = {}
            if mode == 'single_rules':
                rules = _ask_region_rules(root, simpledialog)
                if rules is None:
                    return 0
            report = run_filter_sort(
                APP_ROOT, selected_source=source, book_id=book_id.strip(), stele_id=stele_id.strip(),
                rules_override=rules, progress=lambda s: print(s, flush=True))
            print(_batch_message(report))
            show = messagebox.showwarning if _batch_exit_code(report) or report['report_warnings'] else messagebox.showinfo
            show('Kết quả xử lý box', _batch_message(report), parent=root)
            return _batch_exit_code(report)
        if not normalize_only:
            report = run_school_pipeline(
                APP_ROOT, selected_source=source, book_id=book_id.strip(),
                stele_id=stele_id.strip(), token_provider=lambda: _school_token(root),
                progress=lambda s: print(s, flush=True), include_postprocess=True, include_header=True)
            print(_batch_message(report))
            show = messagebox.showwarning if _batch_exit_code(report) or report['report_warnings'] else messagebox.showinfo
            show('Kết quả pipeline', _batch_message(report), parent=root)
            return _batch_exit_code(report)
        metadata = _normalize_selected_input(
            source, stele_id.strip(), book_id=book_id.strip(), preview=True,
        )
        _print_summary(metadata)
        messagebox.showinfo(
            "Đã chuẩn hóa một ảnh",
            f"JPG:\n{APP_ROOT / metadata['output_jpg']}\n\n"
            f"Trang so sánh:\n{APP_ROOT / metadata['preview_path']}\n\n"
            f"Trang tài liệu: {metadata.get('book_page_number') or 'Chưa được cung cấp'}", parent=root,
        )
        return 0
    except Exception as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        messagebox.showerror("Pipeline đã dừng", str(exc), parent=root)
        return 1
    finally:
        root.destroy()


def _ask_region_rules(root, simpledialog):
    """Optional corrections; default one/All mode requires no box-number input."""
    header = simpledialog.askstring(
        'Trán bia — số box gốc ở 02_ocr',
        'Mở bbox_preview.png ở 02_ocr để xem số box.\n'
        'Nhập box trán bia, ví dụ 1,2,5-7.\n'
        '0 = đã kiểm tra, không có; để trống = chưa rõ/giữ quy tắc đã lưu.', parent=root)
    if header is None:
        return None
    exclude = simpledialog.askstring(
        'Loại box — số box gốc ở 02_ocr',
        'Nhập box ghi chú địa điểm rập bia hoặc box thừa cần loại.\n'
        '0 = đã kiểm tra, không có; để trống = chưa rõ/giữ quy tắc đã lưu.', parent=root)
    if exclude is None:
        return None
    return rules_from_indices(parse_box_indices(header), parse_box_indices(exclude))


def main() -> int:
    if len(sys.argv) == 1:
        return _run_interactive()
    parser = argparse.ArgumentParser(description="Raw PDF/JPG → normalization → school OCR → local box filtering/repair/order")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--all', action='store_true', help='Process all source images in the three raw book folders and retain a detailed report')
    inputs.add_argument("--input", dest="input_path", help="Input PDF/JPG/JPEG path, relative to the Git root")
    parser.add_argument("--normalize-only", action="store_true",
                        help="Only normalize locally; do not call the school API")
    parser.add_argument('--postprocess-only', action='store_true',
                        help='Filter, repair and order existing OCR boxes locally; no API or token')
    parser.add_argument('--header-boxes', help='Original OCR header indices: 1,2,5-7; 0 means no header')
    parser.add_argument('--exclude-boxes', help='Original OCR indices to exclude; 0 confirms none')
    parser.add_argument("--recheck-completed", action="store_true",
                        help="Validate completed images again instead of using the saved work queue")
    parser.add_argument("--book-id", help="Source book ID for a single input")
    parser.add_argument("--stele-id", help="Stele ID for a single input, for example bia_20367")
    parser.add_argument("--page-number", type=int, help="Optional supplied reference page; metadata only, never inferred or used in filenames")
    parser.add_argument("--output", help="<stele-id>.<image-index>.jpg inside the stele's 01_prepare folder")
    parser.add_argument("--render-dpi", type=int, default=300, help="Fallback PDF rendering DPI")
    parser.add_argument("--preview", action="store_true", help="Create a local PDF/JPG comparison page")
    args = parser.parse_args()
    if args.postprocess_only and args.normalize_only:
        parser.error('--postprocess-only cannot be combined with --normalize-only')
    if (args.header_boxes is not None or args.exclude_boxes is not None) and not (args.postprocess_only and args.input_path):
        parser.error('--header-boxes and --exclude-boxes require --postprocess-only --input')
    if args.recheck_completed and (args.normalize_only or args.postprocess_only or not (args.all or args.input_path)):
        parser.error('--recheck-completed requires --all or --input in school OCR mode')
    if not args.all and args.input_path is None:
        if any(value is not None for value in (args.book_id, args.stele_id, args.output, args.page_number)):
            parser.error('Choose --input or --all when providing source options')
        return _run_interactive(normalize_only=args.normalize_only, postprocess_only=args.postprocess_only)
    if args.all and any(value is not None for value in (args.book_id, args.stele_id, args.output, args.page_number)):
        parser.error('--all cannot be combined with --book-id, --stele-id, --output or --page-number')
    if not args.all and (not args.book_id or not args.stele_id):
        parser.error('Single input requires --book-id and --stele-id')
    if not args.normalize_only and (args.output is not None or args.page_number is not None):
        parser.error('--output and --page-number are available with --normalize-only; the full pipeline reuses supplied metadata')
    try:
        if args.postprocess_only:
            report = run_filter_sort(
                APP_ROOT, selected_source=args.input_path, book_id=args.book_id, stele_id=args.stele_id,
                rules_override=rules_from_indices(
                    parse_box_indices(args.header_boxes) if args.header_boxes is not None else None,
                    parse_box_indices(args.exclude_boxes) if args.exclude_boxes is not None else None),
                progress=lambda s: print(s, flush=True))
            print(_batch_message(report))
            return _batch_exit_code(report)
        if not args.normalize_only:
            report = run_school_pipeline(
                APP_ROOT, selected_source=args.input_path, book_id=args.book_id,
                stele_id=args.stele_id, render_dpi=args.render_dpi,
                token_provider=_school_token, progress=lambda s: print(s, flush=True),
                recheck_completed=args.recheck_completed, include_postprocess=True, include_header=True)
            print(_batch_message(report))
            return _batch_exit_code(report)
        if args.all:
            report = run_all(APP_ROOT, render_dpi=args.render_dpi,
                             preview=args.preview, progress=lambda s: print(s, flush=True))
            print(_batch_message(report))
            return _batch_exit_code(report)
        metadata = _normalize_selected_input(
            args.input_path, args.stele_id, args.output, book_id=args.book_id,
            page_number=args.page_number,
            render_dpi=args.render_dpi, preview=args.preview,
        )
    except Exception as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    _print_summary(metadata)
    return 0


if __name__ == "__main__":
    # VS Code's Run button and redirected Windows terminals must print Vietnamese.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
