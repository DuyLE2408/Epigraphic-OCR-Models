"""Batch image normalization with retained per-file reports and safe reruns."""

from collections import Counter
import csv
from datetime import datetime, timezone
from io import StringIO
from itertools import chain
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from src.image_ocr import normalize_single_input, reuse_normalized_output
from src.image_paths import normalization_paths, prepare_folder, same_normalization
from src.page_reference import SUPPORTED_BOOKS, _file_hash, infer_stele_id, supplied_pages, refresh_image_status


class ExistingOutputError(ValueError):
    pass


def _error_reason(exc):
    message = str(exc)
    if isinstance(exc, ExistingOutputError):
        return message
    if isinstance(exc, UnidentifiedImageError):
        return 'File ảnh hỏng hoặc không nhận diện được định dạng. ' + message
    if isinstance(exc, FileExistsError):
        return 'Output đã tồn tại nhưng chưa xác minh được bộ kết quả hợp lệ; không ghi đè. ' + message
    if isinstance(exc, FileNotFoundError):
        return 'Không tìm thấy file/thư mục cần thiết. ' + message
    if 'Expected exactly one PDF page' in message:
        return 'PDF đầu vào không đúng một trang; bước này chỉ xử lý một mặt/trang. ' + message
    if 'content is not JPEG' in message:
        return 'Đuôi file là JPG/JPEG nhưng nội dung không phải JPEG. ' + message
    return message


def _existing_index(app_root: Path, book_id: str, stele_ids=None):
    """Read image metadata only, excluding retained book reference snapshots."""
    index, invalid = {}, {}
    raw = (app_root / 'data/raw' / book_id).resolve()
    paths = (normalization_paths(app_root, book_id) if stele_ids is None else
             chain.from_iterable(normalization_paths(app_root, book_id, stele)
                                 for stele in sorted(stele_ids) if stele is not None))
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(record, dict):
                raise ValueError('JSON kết quả phải là một object')
            source = record.get('source_path') or record.get('input_pdf')
            if not isinstance(source, str):
                continue
            original = (app_root / source).resolve()
            relocated = raw / original.name
            if original.is_relative_to(raw):
                original = relocated  # Support the user's flattened input folders.
            entries = index.setdefault(original, [])
            if not any(same_normalization(existing, record) for _, existing in entries):
                entries.append((path, record))
        except (OSError, ValueError, TypeError) as exc:
            invalid.setdefault(path.parent.name, []).append(f'{path.name}: {exc}')
    return index, invalid


def _validated_existing(source: Path, entries, app_root: Path, book_id: str,
                         max_dim: int | None, render_dpi: int):
    if not entries:
        return None
    if len(entries) != 1:
        raise ExistingOutputError('Có nhiều JSON kết quả cho cùng nguồn; cần kiểm tra trước khi chạy lại.')
    path, data = entries[0]
    source_hash = data.get('source_sha256') or data.get('input_pdf_sha256')
    if source_hash != _file_hash(source):
        raise ExistingOutputError('Ảnh nguồn đã đổi hoặc JSON cũ thiếu checksum; giữ nguyên output, cần kiểm tra trước khi tạo lại.')
    output = data.get('output_jpg')
    if not isinstance(output, str):
        raise ExistingOutputError('JSON cũ thiếu đường dẫn JPG kết quả.')
    image = (app_root / output).resolve()
    allowed = {(app_root / 'data/dataset' / book_id / 'images').resolve(),
               prepare_folder(app_root, book_id, data['stele_id']).resolve()}
    if image.parent not in allowed:
        raise ExistingOutputError('Đường dẫn output trong JSON không thuộc thư mục ảnh của sách.')
    if not image.is_file():
        raise ExistingOutputError('Có JSON nhưng thiếu JPG kết quả; không coi là đã hoàn thành.')
    if data.get('output_jpg_sha256') != _file_hash(image):
        raise ExistingOutputError('Checksum JPG khác JSON cũ; output đã thay đổi hoặc bị hỏng.')
    processing = data.get('processing', {})
    if not isinstance(processing, dict):
        raise ExistingOutputError('Thông tin cấu hình trong JSON cũ không hợp lệ.')
    if processing.get('max_dim') != max_dim:
        raise ExistingOutputError('Cấu hình cạnh ảnh khác kết quả cũ hoặc không được ghi nhận; cần kiểm tra trước khi tạo lại.')
    if max_dim is None and (processing.get('resolution_policy') != 'native'
                            or data.get('resize_scale') != {'x': 1.0, 'y': 1.0}
                            or data.get('image_size_before_resize') != data.get('image_size_after_resize')):
        raise ExistingOutputError('Output cũ chưa xác nhận giữ nguyên độ phân giải; cần tạo lại từ raw.')
    method = data.get('load_method') or data.get('extraction_method')
    if method == 'render' and processing.get('render_dpi') != render_dpi:
        raise ExistingOutputError('DPI render khác kết quả cũ; cần kiểm tra trước khi tạo lại.')
    try:
        with Image.open(image) as decoded:
            decoded.load()
            if (decoded.format != 'JPEG' or decoded.mode != 'RGB'
                    or (max_dim is not None and max(decoded.size) > max_dim)
                    or list(decoded.size) != data.get('image_size_after_resize')):
                raise ExistingOutputError('JPG cũ không đạt định dạng RGB/kích thước chuẩn hóa đã ghi.')
    except OSError as exc:
        raise ExistingOutputError(f'Không đọc được JPG cũ: {exc}') from exc
    return {'output_jpg': output, 'metadata_path': path.relative_to(app_root).as_posix(),
            'image_id': data.get('image_id'), 'image_size': data.get('image_size_after_resize')}


def _atomic_text(path: Path, content: str, encoding='utf-8'):
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding=encoding, newline='', dir=path.parent,
                                prefix='.report_', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _check_unmapped_outputs(app_root: Path, book_id: str, stele_id: str, *, max_dim=None):
    private = app_root / 'data/image_pipeline' / book_id / stele_id
    images = app_root / 'data/dataset' / book_id / 'images'
    stage = prepare_folder(app_root, book_id, stele_id)
    # Dataset exports are independent of the native-resolution processing cache.
    pairs = [(stage, stage)]
    if max_dim is not None:
        pairs.append((images, private))
    for image_folder, metadata_folder in pairs:
        for path in image_folder.glob(f'{stele_id}.*.jpg'):
            if re.fullmatch(rf'{re.escape(stele_id)}\.[0-9]{{3}}\.jpg', path.name):
                if not ((metadata_folder / f'{path.stem}.json').is_file()
                        or (stage / f'{path.stem}.json').is_file()):
                    raise ExistingOutputError(f'Có JPG cũ nhưng thiếu JSON xác định nguồn: {path.name}; không tự tạo ảnh trùng.')
    for folder in (private, stage):
        for path in folder.glob(f'{stele_id}.*.preview.html'):
            name = path.name.removesuffix('.preview.html')
            if not (folder / f'{name}.json').is_file():
                raise ExistingOutputError(f'Có preview cũ nhưng thiếu JSON kết quả: {path.name}; cần kiểm tra.')


STATE_LABELS = {'success': 'Thành công mới', 'skipped_existing': 'Bỏ qua - kết quả hợp lệ',
                'failed': 'Lỗi', 'interrupted': 'Bị ngắt', 'not_run': 'Chưa chạy'}


def _counts(rows):
    count = Counter(row['status'] for row in rows)
    return {'total': len(rows), **{state: count[state] for state in STATE_LABELS}}


def _save_reports(report: dict, folder: Path):
    report['summary'] = _counts(report['files'])
    report['by_book'] = {book: _counts([row for row in report['files'] if row['book_id'] == book])
                         for book in sorted(SUPPORTED_BOOKS)}
    table = StringIO(newline='')
    writer = csv.writer(table)
    writer.writerow(['Sách', 'File nguồn', 'Mã bia', 'Kết quả', 'Nguyên nhân', 'Loại lỗi',
                     'JPG output', 'JSON kết quả', 'Chuẩn hóa', 'OCR', 'Số box',
                     'JSON OCR', 'Ảnh xem box', 'Bước lỗi', 'Xử lý box', 'JSON bước 3',
                     'Lần tự sửa', 'Đề xuất', 'Box cần OCR lại', 'Ảnh đề xuất chưa áp dụng', 'Đuôi box cần đối chiếu',
                     'Box trán bia đề xuất', 'Trạng thái tìm trán bia', 'Báo cáo chọn trán bia'])
    for row in report['files']:
        writer.writerow([row['book_id'], row['source_path'], row.get('stele_id'),
                         STATE_LABELS[row['status']], row.get('reason'), row.get('error_type'),
                         row.get('output_jpg'), row.get('metadata_path'),
                         row.get('normalization_status'), row.get('ocr_status'), row.get('box_count'),
                         row.get('ocr_json'), row.get('bbox_preview'), row.get('failure_stage'),
                         row.get('box_status'), row.get('sorted_json'), row.get('repair_applied'),
                         row.get('repair_proposals'), row.get('ocr_refresh_count'), row.get('proposals_preview'),
                         row.get('tail_review_count'), row.get('header_box_count'), row.get('header_status'),
                         row.get('header_selection')])
    counts = report['summary']
    if report.get('stage') == 'image_pipeline_through_boxes':
        report['box_summary'] = {
            'completed': sum(row.get('box_status') in {'success', 'skipped_existing'} for row in report['files']),
            'applied': sum(row.get('repair_applied', 0) for row in report['files']),
            'proposals': sum(row.get('repair_proposals', 0) for row in report['files']),
            'ocr_refresh': sum(row.get('ocr_refresh_count', 0) for row in report['files']),
            'tail_review': sum(row.get('tail_review_count', 0) for row in report['files']),
        }
    md = [f"# Báo cáo xử lý ảnh — {report['run_id']}", '',
          f"Trạng thái lần chạy: {report['status']}", '',
          f"Tổng {counts['total']} file: thành công mới {counts['success']}; "
          f"bỏ qua hợp lệ {counts['skipped_existing']}; lỗi {counts['failed']}; "
          f"bị ngắt {counts['interrupted']}; chưa chạy {counts['not_run']}.", '',
          '| Sách | Tổng | Mới thành công | Đã có hợp lệ | Lỗi | Bị ngắt | Chưa chạy |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for book, summary in report['by_book'].items():
        md.append('| ' + book + ' | ' + ' | '.join(str(summary[k]) for k in
                  ['total', 'success', 'skipped_existing', 'failed', 'interrupted', 'not_run']) + ' |')
    if report.get('stop_reason'):
        md += ['', 'Lý do dừng: ' + report['stop_reason'],
               'Lần Run tiếp sẽ kiểm tra và dùng lại phần đã hoàn thành.']
    if report.get('box_summary'):
        box = report['box_summary']
        md += ['', f"Bước 3: hoàn thành {box['completed']} ảnh trong lần này; "
               f"tự sửa {box['applied']}; đề xuất {box['proposals']}; box cần OCR lại {box['ocr_refresh']}.",
               f"Đuôi box cần đối chiếu văn bản: {box['tail_review']}.",
               'Ảnh xem box trong báo cáo trỏ tới 03_filter_sort; OCR gốc vẫn giữ ở 02_ocr.']
    if report.get('resume_at_utc'):
        md += ['Thời điểm có thể thử tiếp (UTC): ' + report['resume_at_utc']]
    for row in report['files']:
        if 'header_status' in row:
            md += [f"Trán bia {row.get('image_id')}: {row['header_box_count']} box đề xuất; "
                   f"{row['header_status']}. Cần kiểm tra; không tìm thấy không có nghĩa bia vô đề."]
    if 'api_requests_this_run' in report:
        md += [f"Số lượt API đã gửi trong lần chạy: {report['api_requests_this_run']}."]
    if report.get('progress_file'):
        md += [f"Đã hoàn thành trước lần này: {report['previously_completed']}.",
               f"File tiến độ chung: {report['progress_file']}."]
    md += ['', '## File lỗi hoặc bị ngắt', '', '| File nguồn | Nguyên nhân |', '|---|---|']
    for row in report['files']:
        if row['status'] in {'failed', 'interrupted'}:
            clean = lambda value: str(value).replace('|', '\\|').replace('\n', ' ')
            md.append(f"| {clean(row['source_path'])} | {clean(row['reason'])} |")
    if not any(r['status'] in {'failed', 'interrupted'} for r in report['files']):
        md.append('| — | Không có file lỗi/bị ngắt |')
    md += ['', '## Lỗi thư mục, file loại trừ và cảnh báo', '']
    for item in report['book_errors'] + report['excluded'] + report['report_warnings']:
        md.append('- ' + str(item).replace('\n', ' '))
    if not (report['book_errors'] or report['excluded'] or report['report_warnings']):
        md.append('Không có.')
    md += ['', 'File chi tiết: report.csv. Dữ liệu đầy đủ: report.json.']
    if report.get('progress_file'):
        md.append('Ảnh hoàn thành lấy từ progress.json; dùng --recheck-completed nếu cần kiểm tra lại toàn bộ kết quả.')
    else:
        md.append('Kết quả đã có chỉ được bỏ qua sau khi kiểm tra nguồn, checksum, JPG, JSON và cấu hình.')
    for path, contents, encoding in [(folder / 'report.csv', table.getvalue(), 'utf-8-sig'),
                                      (folder / 'summary.md', '\n'.join(md) + '\n', 'utf-8')]:
        try:
            _atomic_text(path, contents, encoding)
        except OSError as exc:
            warning = f'Không cập nhật được {path.name}: {exc}'
            if warning not in report['report_warnings']:
                report['report_warnings'].append(warning)
    # A JSON checkpoint is mandatory; stop before doing further image work if
    # this fails. CSV/Markdown can temporarily be locked by a viewer.
    _atomic_text(folder / 'report.json', json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def run_all(app_root: Path, *, max_dim=None, render_dpi=300, preview=True, progress=print,
            ocr_processor=None, box_processor=None, selected_source=None, progress_store=None) -> dict:
    """Normalize every source of the three books; never overwrite old image outputs."""
    if max_dim is not None and (isinstance(max_dim, bool) or not isinstance(max_dim, int) or max_dim < 1):
        raise ValueError('max_dim must be a positive integer or None')
    if isinstance(render_dpi, bool) or not isinstance(render_dpi, int) or render_dpi < 72:
        raise ValueError('render_dpi must be an integer of at least 72')
    app_root = Path(app_root).resolve()
    started = datetime.now(timezone.utc)
    run_id = started.strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    folder = app_root / 'data/image_pipeline/run_reports' / run_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': '1.1', 'run_id': run_id,
              'stage': ('image_pipeline_through_boxes' if box_processor else
                        'image_normalization_and_school_ocr' if ocr_processor else 'image_normalization'),
              'status': 'running', 'started_at_utc': started.isoformat(), 'finished_at_utc': None,
              'report_dir': str(folder), 'files': [], 'excluded': [], 'book_errors': [],
              'stop_reason': None, 'resume_at_utc': None,
              'previously_completed': 0,
              'report_warnings': [], 'settings': {'max_dim': max_dim, 'render_dpi': render_dpi,
                                                'resolution_policy': 'native' if max_dim is None else 'limited',
                                                'preview': preview, 'overwrite': False}}
    provided, existing, invalid_metadata = {}, {}, {}
    if progress_store is not None:
        report['progress_file'] = str(progress_store.path)
    selected_source = Path(selected_source).resolve() if selected_source is not None else None
    books = sorted(SUPPORTED_BOOKS)
    if selected_source is not None:
        books = [book for book in books
                 if selected_source.parent == (app_root / 'data/raw' / book).resolve()]
        if not books or not selected_source.is_file():
            raise ValueError('Selected input must be an existing file directly inside a raw book folder.')
    for book_id in books:
        raw = app_root / 'data/raw' / book_id
        if not raw.is_dir():
            report['book_errors'].append(f'{book_id}: thiếu thư mục {raw}')
            continue
        try:
            sources = [selected_source] if selected_source is not None else sorted(raw.iterdir())
        except OSError as exc:
            report['book_errors'].append(f'{book_id}: không đọc được thư mục nguồn: {exc}')
            continue
        for source in sources:
            if not source.is_file() or source.suffix.lower() not in {'.jpg', '.jpeg', '.pdf'}:
                report['excluded'].append(f'{book_id}/{source.name}: không phải file ảnh/PDF đầu vào trực tiếp')
                continue
            try:
                stele_id = infer_stele_id(source)
            except ValueError:
                if source.suffix.lower() == '.pdf':
                    report['excluded'].append(f'{book_id}/{source.name}: PDF không có mã bia đầu tên, không xử lý sách đầy đủ')
                    continue
                stele_id = None
            relative = source.relative_to(app_root).as_posix()
            if (progress_store is not None and selected_source is None
                    and progress_store.should_skip(relative, source)):
                report['previously_completed'] += 1
                continue
            report['files'].append({'book_id': book_id, 'source_path': relative,
                                    'stele_id': stele_id, 'status': 'not_run', 'reason': 'Chưa đến lượt xử lý'})
        active = {row['stele_id'] for row in report['files'] if row['book_id'] == book_id}
        if progress_store is not None and not active:
            continue
        try:
            provided.update({r['source_path']: r for r in supplied_pages(app_root, book_id, progress=progress)})
            existing[book_id], invalid_metadata[book_id] = _existing_index(
                app_root, book_id, active if progress_store is not None else None)
        except (OSError, ValueError) as exc:
            report['book_errors'].append(f'{book_id}: không đọc được dữ liệu hỗ trợ: {exc}')
            existing[book_id], invalid_metadata[book_id] = {}, {}
    if progress_store is not None:
        progress_store.prepare_queue(report['files'], full_scan=selected_source is None, run_id=run_id)
        progress(f"Tiến độ: {report['previously_completed']} ảnh đã hoàn thành; "
                 f"chỉ xử lý {len(report['files'])} ảnh còn lại.")
    _save_reports(report, folder)
    progress(f"All: {len(report['files'])} ảnh/PDF nguồn. Báo cáo: {folder}")
    try:
        for index, row in enumerate(report['files'], 1):
            source, book_id = app_root / row['source_path'], row['book_id']
            stage = 'normalization'
            try:
                if row['stele_id'] is None:
                    infer_stele_id(source)  # Raise the filename error for this JPG.
                errors = invalid_metadata.get(book_id, {}).get(row['stele_id'], [])
                if errors:
                    raise ExistingOutputError('JSON kết quả cũ không đọc được: ' + '; '.join(errors))
                prior = _validated_existing(source, existing.get(book_id, {}).get(source.resolve(), []),
                                            app_root, book_id, max_dim, render_dpi)
                if prior:
                    row.update(prior, status='skipped_existing', reason='Nguồn, JPG, JSON và cấu hình còn hợp lệ; giữ kết quả cũ.')
                    row['normalization_status'] = 'skipped_existing'
                    metadata = json.loads((app_root / prior['metadata_path']).read_text(encoding='utf-8'))
                    metadata = reuse_normalized_output(metadata, app_root, preview=preview)
                    row.update(output_jpg=metadata['output_jpg'], metadata_path=metadata['metadata_path'],
                               image_id=metadata['image_id'])
                else:
                    _check_unmapped_outputs(app_root, book_id, row['stele_id'], max_dim=max_dim)
                    metadata = normalize_single_input(source, row['stele_id'], book_id=book_id,
                        provided_page=provided.get(row['source_path']), app_root=app_root,
                        max_dim=max_dim, render_dpi=render_dpi, preview=preview)
                    row.update(status='success', reason='Đã tạo JPG và JSON kết quả.',
                               output_jpg=metadata['output_jpg'], metadata_path=metadata['metadata_path'],
                               image_id=metadata['image_id'], image_size=metadata['image_size_after_resize'])
                    row['normalization_status'] = 'success'
                if ocr_processor is not None:
                    stage = 'ocr'
                    ocr = ocr_processor(metadata)
                    row.update(ocr)
                    if ocr['ocr_status'] == 'success':
                        row.update(status='success', reason=f"Đã lưu OCR và ảnh xem {ocr['box_count']} box; cần kiểm tra box.")
                    else:
                        row.update(reason='Đã có chuẩn hóa và OCR hợp lệ; không gửi lại ảnh.')
                if box_processor is not None:
                    stage = 'box_filter_sort'
                    progress(f"[Bước 3] {metadata['image_id']}: lọc / tự sửa / sắp box.")
                    post = box_processor(metadata)
                    row.update(ocr_box_count=row.get('box_count'), ocr_bbox_preview=row.get('bbox_preview'),
                               box_status=post['status'])
                    row.update({key: post[key] for key in ('box_count', 'sorted_json', 'bbox_preview',
                               'repair_applied', 'repair_proposals', 'ocr_refresh_count',
                               'order_status', 'warnings')})
                    row['proposals_preview'] = post.get('proposals_preview')
                    row['tail_review_count'] = post.get('tail_review_count', 0)
                    if post['status'] == 'success':
                        row['status'] = 'success'
                    row['reason'] = (f"Đã hoàn thành đến bước 3: {post['box_count']} box; "
                                     f"tự sửa {post['repair_applied']}; đề xuất {post['repair_proposals']}; "
                                     f"box cần OCR lại {post['ocr_refresh_count']}.")
            except KeyboardInterrupt:
                row.update(status='interrupted', error_type='KeyboardInterrupt', reason='Người dùng ngắt khi đang xử lý file này; kiểm tra output trước khi chạy lại.')
                if progress_store is not None:
                    progress_store.record(row)
                raise
            except Exception as exc:
                row.update(status='failed', failure_stage=stage,
                           error_type=type(exc).__name__, reason=_error_reason(exc))
                if getattr(exc, 'stop_pipeline', False):
                    resume = getattr(exc, 'resume_at', None)
                    report.update(status='paused' if resume else 'stopped',
                                  stop_reason=str(exc), resume_at_utc=resume)
                    if resume:
                        row.update(status='not_run', ocr_status='waiting_quota')
                    if progress_store is not None:
                        progress_store.record(row)
                    progress('Đã lưu tiến độ và dừng: ' + str(exc))
                    break
            if progress_store is not None:
                progress_store.record(row)
            _save_reports(report, folder)
            progress(f"[{index}/{len(report['files'])}] {STATE_LABELS[row['status']]}: {row['source_path']} — {row['reason']}")
        if report['status'] == 'running':
            report['status'] = 'completed'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        progress('Đã ngắt All; giữ báo cáo và các kết quả đã hoàn thành.')
    finally:
        for book_id in books:
            if progress_store is not None:
                continue  # Full image-status hashing is an explicit recheck, not part of fast resume.
            if (app_root / 'data/raw' / book_id).is_dir():
                try:
                    refresh_image_status(app_root, book_id, progress=progress)
                except Exception as exc:
                    report['report_warnings'].append(f'{book_id}: không cập nhật được image_status.csv: {exc}')
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        gate = getattr(getattr(ocr_processor, 'client', None), 'gate', None)
        calls = getattr(gate, 'calls', None)
        if type(calls) is int:
            report['api_requests_this_run'] = calls
        if progress_store is not None:
            progress_store.finish(report)
        _save_reports(report, folder)
    return report
