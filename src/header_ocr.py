"""Full-image horizontal OCR and provisional header selection; never crop/upload regions."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from uuid import uuid4

from src.image_ocr import _write_new_files
from src.image_paths import combined_ocr_paths, ocr_folder
from src.page_reference import _file_hash
from src.school_ocr import OCR_PARAMETERS, SchoolAPIError, SchoolOCR, _json, _preview


HEADER_PARAMETERS = {**OCR_PARAMETERS, 'ocr_id': 4, 'reading_direction': 2, 'epitaph': 0}
HEADER_PASS_FOLDER = 'header/general'
SELECTION_VERSION = '1.1'
HAN = re.compile('[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U000323af]')


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _rect(box):
    return [min(p[0] for p in box['bbox']), min(p[1] for p in box['bbox']),
            max(p[0] for p in box['bbox']), max(p[1] for p in box['bbox'])]


def _inside(rect, region):
    return region[0] <= rect[0] < rect[2] <= region[2] and region[1] <= rect[1] < rect[3] <= region[3]


def _inside_automatic_band(rect, region, column_width):
    """An estimated body top is not an exact boundary through every title glyph."""
    if _inside(rect, region):
        return True
    height = rect[3] - rect[1]
    return (height > 0 and region[0] <= rect[0] < rect[2] <= region[2]
            and region[1] <= rect[1] < region[3]
            and (rect[1] + rect[3]) / 2 < region[3]
            and (region[3] - rect[1]) / height >= 0.6
            and rect[3] - region[3] <= column_width)


def infer_header_region(boxes, image_size):
    """Estimate a search band above a dense group of vertical columns, not a title label."""
    vertical = []
    for box in boxes:
        x1, y1, x2, y2 = _rect(box)
        if x2 - x1 >= 3 and y2 - y1 >= 4 * (x2 - x1):
            vertical.append(box)
    if len(vertical) < 3:
        return None
    unit = median(_rect(b)[2] - _rect(b)[0] for b in vertical)
    groups = []
    for box in sorted(vertical, key=lambda b: sum(_rect(b)[::2])):
        center = sum(_rect(box)[::2]) / 2
        if not groups or center - sum(_rect(groups[-1][-1])[::2]) / 2 > 3.5 * unit:
            groups.append([])
        groups[-1].append(box)
    group = max(groups, key=lambda g: (len(g), sum(_rect(b)[3] - _rect(b)[1] for b in g)))
    centers = sorted(sum(_rect(b)[::2]) / 2 for b in group)
    distinct = [centers[0]]
    for center in centers[1:]:
        if center - distinct[-1] >= 0.6 * unit:
            distinct.append(center)
    if len(distinct) < 3:
        return None
    unit = median(_rect(b)[2] - _rect(b)[0] for b in group)
    tops = sorted(_rect(b)[1] for b in group)
    top = tops[(len(tops) - 1) // 4]
    width, height = image_size
    region = [max(0, min(_rect(b)[0] for b in group) - 2 * unit),
              max(0, top - max(5 * unit, 0.12 * height)),
              min(width, max(_rect(b)[2] for b in group) + 2 * unit), top]
    return {'region': region, 'body_column_width': unit, 'body_columns': len(distinct)}


def _config(folder, size):
    path = folder / 'header_regions.json'
    config = _json(path) if path.exists() else {'regions': None}
    if set(config) != {'regions'}:
        raise SchoolAPIError('header_regions.json chỉ hỗ trợ khóa regions.')
    regions = config['regions']
    if regions is not None:
        if not isinstance(regions, list) or not regions:
            raise SchoolAPIError('regions phải là danh sách vùng không rỗng, hoặc null để tự đề xuất.')
        for r in regions:
            if (not isinstance(r, list) or len(r) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in r)
                    or not 0 <= r[0] < r[2] <= size[0] or not 0 <= r[1] < r[3] <= size[1]):
                raise SchoolAPIError('Vùng trán bia phải là [x1,y1,x2,y2] trên ảnh 01_prepare.')
    return config


def select_header_boxes(body, horizontal, size, regions=None):
    """Keep only plausible horizontal-pass headers; uncertain text remains in raw results."""
    from src.box_postprocess import _valid_polygon
    estimate = infer_header_region(body, size) if regions is None else None
    search = regions if regions is not None else [estimate['region']] if estimate else []
    selected, rejected = [], []
    for original in horizontal:
        box = deepcopy(original)
        rect = _rect(box)
        width, height = rect[2] - rect[0], rect[3] - rect[1]
        reason = None
        if not _valid_polygon(box['bbox']) or not _inside(rect, [0, 0, *size]):
            reason = 'invalid_or_outside_image'
        elif not search:
            reason = 'header_region_unknown'
        elif not any((_inside(rect, r) if regions is not None else
                      _inside_automatic_band(rect, r, estimate['body_column_width'])) for r in search):
            reason = 'outside_header_search_region'
        elif regions is None:
            unit = estimate['body_column_width']
            if not (0.7 * unit <= width and 1.15 * unit <= height <= 5 * unit
                    and width >= 0.65 * height):
                reason = 'not_title_sized_horizontal_text'
            elif not HAN.search(box.get('sino_nom_ocr') or ''):
                reason = 'no_han_text_for_automatic_selection'
        if reason:
            rejected.append({'ocr_index': original['ocr_index'], 'bbox': original['bbox'], 'reason': reason})
        else:
            box.update(region='header', is_header=True, region_source='horizontal_header_candidate',
                       source_ocr_pass='horizontal', source_ocr_index=original['ocr_index'])
            selected.append(box)
    # Keep the first of nearly identical overlapping detections, preserving the alternatives in raw OCR.
    unique = []
    for box in selected:
        if any(_overlap_fraction(box, prior) >= 0.85 for prior in unique):
            rejected.append({'ocr_index': box['ocr_index'], 'bbox': box['bbox'], 'reason': 'overlapping_header_candidate'})
        else:
            unique.append(box)
    report = {'selection_version': SELECTION_VERSION,
              'region_source': 'configured' if regions is not None else 'automatic_geometry',
              'search_regions': search, 'body_estimate': estimate,
              'automatic_boundary_overlap': ({'minimum_height_inside_fraction': 0.6,
                  'maximum_pixels_below_body_top': estimate['body_column_width'],
                  'center_must_be_above_body_top': True} if estimate else None),
              'status': 'candidates_found' if unique else 'not_found' if search else 'region_unknown',
              'selected_horizontal_indices': [b['source_ocr_index'] for b in unique],
              'rejected': rejected, 'review_required': True}
    return unique, report


def _overlap_fraction(a, b):
    a, b = _rect(a), _rect(b)
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return area / max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def _inputs(folder):
    header_folder = folder / HEADER_PASS_FOLDER
    body, header = _json(folder / 'ocr.json'), _json(header_folder / 'ocr.json')
    for key in ('image_id', 'book_id', 'image_path', 'image_sha256', 'image_size', 'coordinate_space'):
        if body.get(key) != header.get(key):
            raise SchoolAPIError('Hai lượt OCR không dùng cùng một ảnh và hệ tọa độ.')
    if body.get('parameters') != OCR_PARAMETERS or header.get('parameters') != HEADER_PARAMETERS:
        raise SchoolAPIError('Tham số hai lượt OCR không đúng dọc/ngang.')
    config = _config(folder, body['image_size'])
    sources = {'body': _file_hash(folder / 'ocr.json'), 'header': _file_hash(header_folder / 'ocr.json'),
               'body_complete': _file_hash(folder / 'complete.json'),
               'header_complete': _file_hash(header_folder / 'complete.json'),
               'config': hashlib.sha256(_encoded(config)).hexdigest(), 'selection_version': SELECTION_VERSION}
    return body, header, config, sources


def _check_outputs(paths, marker):
    for key, path in paths.items():
        if key == 'complete':
            continue
        if not path.is_file() or _file_hash(path) != marker.get('sha256', {}).get(key):
            raise SchoolAPIError(f'Kết quả ghép trán bia thiếu/đã sửa: {path.name}; giữ lại để kiểm tra.')


def header_result_is_current(root, book, stele, image_id):
    try:
        folder = ocr_folder(root, book, stele, image_id)
        _, _, _, sources = _inputs(folder)
        paths = combined_ocr_paths(folder)
        marker = _json(paths['complete'])
        for pass_folder in (folder, folder / HEADER_PASS_FOLDER):
            completed = _json(pass_folder / 'complete.json')
            document = _json(pass_folder / 'ocr.json')
            if completed.get('binding') != {k: document[k] for k in ('image_id', 'image_sha256', 'parameters')}:
                return False
            for key, name in (('raw', 'ocr_raw.json'), ('boxes', 'ocr.json'), ('preview', 'bbox_preview.png')):
                if _file_hash(pass_folder / name) != completed.get('sha256', {}).get(key):
                    return False
        _check_outputs(paths, marker)
        return marker.get('sources') == sources
    except (OSError, ValueError, TypeError, KeyError, SchoolAPIError):
        return False


class HeaderAwareOCR:
    def __init__(self, root, client, *, progress=print):
        self.root, self.client, self.progress = Path(root).resolve(), client, progress
        self.body = SchoolOCR(root, client, progress=progress)
        self.header = SchoolOCR(root, client, progress=progress, parameters=HEADER_PARAMETERS,
                                pass_name=HEADER_PASS_FOLDER)

    def __call__(self, metadata):
        folder = ocr_folder(self.root, metadata['book_id'], metadata['stele_id'], metadata['image_id'])
        _config(folder, metadata['image_size_after_resize'])
        self.body(metadata)
        self.header(metadata)
        body, horizontal, config, sources = _inputs(folder)
        if not horizontal['boxes']:
            self.progress(f"{body['image_id']}: API ngang không trả box nào; "
                          'không phải do bộ lọc trán bia loại mất box.')
        paths = combined_ocr_paths(folder)
        if paths['complete'].exists():
            marker = _json(paths['complete'])
            _check_outputs(paths, marker)
            if marker.get('sources') == sources:
                saved = _json(paths['boxes'])
                return self._summary(paths, saved, 'skipped_existing')
            history = folder / '_combined_history' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid4().hex[:8])
            if (not history.resolve().is_relative_to(self.root)
                    or any(path.resolve().parent != folder.resolve() for path in paths.values())):
                raise SchoolAPIError('Đường dẫn lịch sử kết quả ghép không hợp lệ.')
            history.mkdir(parents=True, exist_ok=False)
            for path in paths.values():
                path.rename(history / path.name)
        headers, selection = select_header_boxes(body['boxes'], horizontal['boxes'], body['image_size'], config['regions'])
        boxes, superseded = [], []
        for original in body['boxes']:
            if (any(_inside(_rect(original), region) for region in selection['search_regions'])
                    and any(_overlap_fraction(original, h) >= 0.75 for h in headers)):
                superseded.append(original['ocr_index'])
                continue
            box = deepcopy(original)
            box.update(source_ocr_pass='vertical', source_ocr_index=original['ocr_index'])
            boxes.append(box)
        next_index = max((b['ocr_index'] for b in body['boxes']), default=0)
        for index, box in enumerate(headers, next_index + 1):
            box.update(ocr_index=index, patch_id=f"{body['image_id']}.{index:03d}")
            boxes.append(box)
        if len(boxes) > 999:
            raise SchoolAPIError('Quá 999 box sau ghép; cần kiểm tra trước khi cấp số patches.')
        selection.update(superseded_vertical_indices=superseded, header_box_count=len(headers))
        parameters = {'vertical': OCR_PARAMETERS, 'horizontal': HEADER_PARAMETERS, 'selection_version': SELECTION_VERSION}
        binding = {k: body[k] for k in ('image_id', 'image_sha256')}
        binding['parameters'] = parameters
        saved = {**body, **binding, 'schema_version': 'school-ocr-header-1.0',
                 'boxes': boxes, 'header_status': selection['status'], 'header_box_count': len(headers),
                 'header_review_required': True}
        image = self.root / body['image_path']
        payloads = {paths['boxes']: _encoded(saved), paths['selection']: _encoded(selection),
                    paths['raw']: _encoded({'kind': 'local_combination_not_api_response', 'sources': sources,
                        'vertical_raw': (folder / 'ocr_raw.json').relative_to(self.root).as_posix(),
                        'horizontal_raw': (folder / HEADER_PASS_FOLDER / 'ocr_raw.json').relative_to(self.root).as_posix()}),
                    paths['preview']: _preview(image, boxes)}
        for path, content in payloads.items():
            if path.exists() and path.read_bytes() != content:
                raise SchoolAPIError(f'File ghép chưa hoàn tất có nội dung khác: {path.name}; không ghi đè.')
        _write_new_files({p: data for p, data in payloads.items() if not p.exists()})
        if _file_hash(image) != body['image_sha256'] or _inputs(folder)[3] != sources:
            raise SchoolAPIError('Nguồn thay đổi khi ghép trán bia; chưa ghi hoàn thành.')
        _write_new_files({paths['complete']: _encoded({'binding': binding, 'sources': sources,
                         'sha256': {key: _file_hash(path) for key, path in paths.items() if key != 'complete'}})})
        self.progress(f"{body['image_id']}: lượt ngang {len(horizontal['boxes'])} box; "
                      f"chọn {len(headers)} box trán bia ({selection['status']}); cần kiểm tra.")
        return self._summary(paths, saved, 'success')

    def _summary(self, paths, saved, status):
        folder = paths['boxes'].parent
        return {'ocr_status': status, 'box_count': len(saved['boxes']), 'text_count': len(saved['boxes']),
                'ocr_json': paths['boxes'].relative_to(self.root).as_posix(),
                'ocr_raw_json': (folder / 'ocr_raw.json').relative_to(self.root).as_posix(),
                'header_ocr_raw_json': (folder / HEADER_PASS_FOLDER / 'ocr_raw.json').relative_to(self.root).as_posix(),
                'combined_sources_json': paths['raw'].relative_to(self.root).as_posix(),
                'bbox_preview': paths['preview'].relative_to(self.root).as_posix(),
                'header_box_count': saved['header_box_count'], 'header_status': saved['header_status'],
                'header_selection': paths['selection'].relative_to(self.root).as_posix(),
                'header_review_required': True}
