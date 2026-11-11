"""Supplied page metadata and image status only. No PDF search or OCR."""

import csv
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from zipfile import ZipFile, BadZipFile
import xml.etree.ElementTree as ET

from src.image_paths import normalization_paths, prepare_folder, same_normalization


EXCEL_BOOK = 'van_bia_thoi_tran'
SUPPORTED_BOOKS = {'di_san_han_nom_huyen_dai_loc', EXCEL_BOOK,
                   'van_khac_han_nom_thang_long_ha_noi'}


def _file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def infer_stele_id(source: Path) -> str:
    """Use the first code; later ampersand codes are aliases, not extra images."""
    match = re.match(r'^([0-9]+)(?=[&.-]|$)', source.stem)
    if not match or int(match.group(1)) < 1:
        raise ValueError('Source filename must start with a stele code, e.g. 07804.jpg or 20367.pdf.')
    return f'bia_{int(match.group(1))}'


def collect_sources(app_root: Path, book_id: str) -> list[dict]:
    records = []
    for source in sorted((app_root / 'data/raw' / book_id).iterdir()):
        if not source.is_file() or source.suffix.lower() not in {'.jpg', '.jpeg', '.pdf'}:
            continue
        try:
            stele_id = infer_stele_id(source)
        except ValueError:
            continue  # Full book PDFs and unrelated files have no source stele prefix.
        records.append({
            'source_path': source.relative_to(app_root).as_posix(),
            'stele_id': stele_id, 'code': stele_id.removeprefix('bia_'),
            'book_page_number': None, 'page_scope': None, 'page_number_source': None,
        })
    return records


def _excel_rows(path: Path):
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in si.findall('.//m:t', ns))
                       for si in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        for row in sheet.findall('.//m:sheetData/m:row', ns):
            cells = {}
            for cell in row.findall('m:c', ns):
                value = cell.find('m:v', ns)
                text = value.text if value is not None else ''.join(t.text or '' for t in cell.findall('.//m:t', ns))
                if cell.get('t') == 's':
                    text = strings[int(text)]
                cells[re.sub(r'\d', '', cell.get('r'))] = text or ''
            yield int(row.get('r')), cells


def supplied_pages(app_root: Path, book_id: str, *, assignment: Path | None = None,
                   progress=print) -> list[dict]:
    """Only a supplied Excel column can populate pages; unknowns stay None.

    Multi-face entries retain the supplied article reference page, explicitly
    scoped to the stele entry. It never names or counts source images.
    """
    records = collect_sources(app_root, book_id)
    if book_id != EXCEL_BOOK:
        return records
    assignment = Path(assignment) if assignment else app_root.parent.parent / 'project_material/Nhom4/DanhSach.xlsx'
    if not assignment.is_file():
        progress('Supplied page workbook unavailable; page metadata will remain empty.')
        return records
    try:
        matches_by_source = {}
        for row_number, cells in _excel_rows(assignment):
            label = re.sub(r'\s', '', cells.get('H', ''))
            page = cells.get('G', '')
            if not re.fullmatch(r'[0-9]+(?:-[0-9]+)*', label) or not page.isdigit() or int(page) < 1:
                continue
            codes = {str(int(code)) for code in label.split('-')}
            correction = codes == {'4997', '5001'}
            if correction or codes == {'4997', '5000'}:
                codes = {'4997', '4998', '4999', '5000'}  # User-confirmed group.
            for record in records:
                if record['code'] in codes:
                    matches_by_source.setdefault(record['source_path'], []).append({
                        'book_page_number': int(page), 'page_scope': 'stele_entry',
                        'page_number_source': 'provided_excel',
                        'assignment_path': str(assignment), 'excel_row': row_number,
                        'user_confirmed_code_correction': correction,
                    })
        for record in records:
            matches = matches_by_source.get(record['source_path'], [])
            if len(matches) == 1:
                record.update(matches[0])
            elif len(matches) > 1:
                progress(f"Conflicting supplied pages for {record['stele_id']}; leaving page empty.")
    except (OSError, BadZipFile, ET.ParseError, ValueError, KeyError):
        progress('Cannot read supplied page workbook; page metadata will remain empty.')
    return records


def get_provided_page(app_root: Path, book_id: str, source: Path, **kwargs) -> dict:
    source = source.resolve()
    matches = [r for r in supplied_pages(app_root, book_id, **kwargs)
               if (app_root / r['source_path']).resolve() == source]
    if len(matches) != 1:
        raise ValueError('Input has no unique entry in the raw source folders.')
    return matches[0]


def refresh_image_status(app_root: Path, book_id: str, *, progress=print) -> Path:
    """Report current supplied pages and normalization status without old OCR guesses."""
    app_root = Path(app_root)
    records = supplied_pages(app_root, book_id, progress=progress)
    folder = app_root / 'data/image_pipeline' / book_id
    outputs = {}
    images = (app_root / 'data/dataset' / book_id / 'images').resolve()
    for metadata_path in normalization_paths(app_root, book_id):
        try:
            data = json.loads(metadata_path.read_text(encoding='utf-8'))
            source = data.get('source_path') or data.get('input_pdf')
            output = data.get('output_jpg')
            if not source or not output:
                continue
            image, original = (app_root / output).resolve(), (app_root / source).resolve()
            raw_book = (app_root / 'data/raw' / book_id).resolve()
            relocated = raw_book / original.name
            if original.is_relative_to(raw_book) and relocated.is_file():
                original = relocated
            stele_id = data.get('stele_id', metadata_path.parent.name)
            allowed = {images, prepare_folder(app_root, book_id, stele_id).resolve()}
            good = (image.parent in allowed and image.is_file()
                    and data.get('output_jpg_sha256') == _file_hash(image))
            source_hash = data.get('source_sha256') or data.get('input_pdf_sha256')
            unchanged = original.is_file() and source_hash == _file_hash(original)
            state = 'ok' if good and unchanged else 'source_changed' if good else 'output_invalid'
            matches = outputs.setdefault(original, [])
            if not any(same_normalization(existing, data) for _, _, existing in matches):
                matches.append((output, state, data))
        except (OSError, ValueError, TypeError):
            continue
    table = StringIO(newline='')
    writer = csv.writer(table)
    writer.writerow(['Mã bia / nhóm', 'Mã ảnh nguồn', 'File nguồn', 'Trang được cung cấp',
                     'Phạm vi số trang', 'Nguồn số trang', 'Trạng thái ảnh', 'File ảnh output'])
    for record in records:
        found = outputs.get((app_root / record['source_path']).resolve(), [])
        if not found:
            state = 'Chưa chuẩn hóa'
        elif len(found) > 1:
            state = 'Có nhiều output - cần kiểm tra'
        else:
            state = {'ok': 'Đã chuẩn hóa', 'source_changed': 'Ảnh nguồn đã đổi/thiếu - cần xử lý lại',
                     'output_invalid': 'Thiếu/lỗi output'}[found[0][1]]
            metadata = found[0][2]
            if metadata.get('page_number_source') == 'provided_explicit':
                record.update(book_page_number=metadata.get('book_page_number'),
                              page_scope='user_provided', page_number_source='provided_explicit')
        writer.writerow([record['stele_id'], record.get('code'), record['source_path'],
                         record['book_page_number'], record['page_scope'], record['page_number_source'],
                         state, ' | '.join(output for output, _, _ in found)])
    folder.mkdir(parents=True, exist_ok=True)
    path, temporary = folder / 'image_status.csv', None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                dir=folder, prefix='.image_status_', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(table.getvalue())
        os.replace(temporary, path)
    except OSError:
        progress(f'Cannot update image status CSV; close it in Excel and retry: {path}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
