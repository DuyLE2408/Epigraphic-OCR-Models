"""Filter, repair and order derived boxes while preserving original OCR files."""

from copy import deepcopy
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import re
from statistics import median
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from src.image_batch import _atomic_text, _counts
from src.image_ocr import _write_new_files
from src.image_paths import (filter_sort_folder, normalization_paths, ocr_folder, prepare_folder,
                             combined_ocr_paths, preferred_ocr_path)
from src.page_reference import SUPPORTED_BOOKS, _file_hash, infer_stele_id
from src.school_ocr import _single_run
from src.box_repair import REPAIR_VERSION, repair_boxes


ALGORITHM_VERSION = "2.4"
DEFAULT_RULES = {
    "header_present": None, "header_boxes": [], "exclude_boxes": [],
    "header_regions": [], "exclude_regions": [], "annotation_checked": False,
    "auto_repair": True,
    'tighten_boxes': True,
}


def parse_box_indices(value):
    """Original OCR indices: '1,2,5-7'; 0 means confirmed none; blank means unknown."""
    if not value.strip():
        return None
    if value.strip() == "0":
        return []
    values = set()
    for part in re.split(r"[,\s]+", value.strip()):
        match = re.fullmatch(r"([1-9][0-9]*)(?:-([1-9][0-9]*))?", part)
        if not match:
            raise ValueError("Dùng số box OCR, ví dụ 1,2,5-7; nhập 0 nếu không có.")
        first, last = int(match[1]), int(match[2] or match[1])
        if not first <= last <= 999:
            raise ValueError("Số box phải trong khoảng 1–999.")
        values.update(range(first, last + 1))
    return sorted(values)


def rules_from_indices(header=None, exclude=None):
    rules = {}
    if header is not None:
        rules.update(header_boxes=header, header_present=bool(header))
    if exclude is not None:
        rules.update(exclude_boxes=exclude, annotation_checked=True)
    return rules


def _rect(box):
    points = box["bbox"]
    return (min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points))


def _valid_polygon(points):
    if (not isinstance(points, list) or len(points) != 4
            or any(not isinstance(p, (list, tuple)) or len(p) != 2
                   or any(isinstance(v, bool) or not isinstance(v, (int, float))
                          or not math.isfinite(v) for v in p) for p in points)):
        return False
    crosses = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(v > 0 for v in crosses) or all(v < 0 for v in crosses)


def _contained(rect, region):
    return region[0] <= rect[0] <= rect[2] <= region[2] and region[1] <= rect[1] <= rect[3] <= region[3]


def _bands(boxes, axis):
    """Group aligned fragments; never merge their polygons or OCR strings."""
    groups = []
    for box in sorted(boxes, key=lambda b: ((_rect(b)[axis] + _rect(b)[axis + 2]) / 2,
                                             b["ocr_index"])):
        rect = _rect(box)
        center, extent = (rect[axis] + rect[axis + 2]) / 2, rect[axis + 2] - rect[axis]
        choices = []
        for index, group in enumerate(groups):
            perpendicular = 1 - axis
            # Parallel full-height columns must never become one reading band
            # merely because their bounding rectangles overlap in x after skew.
            if any(max(0, min(rect[perpendicular + 2], _rect(b)[perpendicular + 2])
                       - max(rect[perpendicular], _rect(b)[perpendicular]))
                   > 0.2 * min(rect[perpendicular + 2] - rect[perpendicular],
                               _rect(b)[perpendicular + 2] - _rect(b)[perpendicular]) for b in group):
                continue
            centers = [(_rect(b)[axis] + _rect(b)[axis + 2]) / 2 for b in group]
            widths = [_rect(b)[axis + 2] - _rect(b)[axis] for b in group]
            distance = abs(center - median(centers))
            if distance <= max(1, 0.65 * min(extent, median(widths))):
                choices.append((distance, index))
        if choices:
            groups[min(choices)[1]].append(box)
        else:
            groups.append([box])
    return sorted(groups, key=lambda group: median((_rect(b)[axis] + _rect(b)[axis + 2]) / 2 for b in group))


def sort_boxes(boxes):
    """Header rows right-to-left, then body columns left-to-right/top-to-bottom."""
    boxes = deepcopy(boxes)
    for index, box in enumerate(boxes, 1):
        if not _valid_polygon(box.get("bbox")):
            raise ValueError("Không thể sắp box có tọa độ không hợp lệ.")
        box.setdefault("ocr_index", index)
    headers = [b for b in boxes if b.get("region") == "header" or b.get("is_header") is True]
    body = [b for b in boxes if b not in headers]
    ordered = []
    for row_number, row in enumerate(_bands(headers, 1), 1):
        for box in sorted(row, key=lambda b: (-(_rect(b)[0] + _rect(b)[2]) / 2,
                                              _rect(b)[1], b["ocr_index"])):
            box["reading_group"] = f"header_row_{row_number:03d}"
            ordered.append(box)
    for column_number, column in enumerate(_bands(body, 0), 1):
        for box in sorted(column, key=lambda b: (_rect(b)[1], _rect(b)[0], b["ocr_index"])):
            box["reading_group"] = f"body_column_{column_number:03d}"
            ordered.append(box)
    for index, box in enumerate(ordered, 1):
        box["reading_order"] = index
    return ordered


def filter_and_sort(boxes, image_size, rules=None, *, image=None):
    rules = {**deepcopy(DEFAULT_RULES), **(rules or {})}
    if set(rules) != set(DEFAULT_RULES):
        raise ValueError("rules.json có khóa không được hỗ trợ.")
    if (rules["header_present"] is not None and type(rules["header_present"]) is not bool
            or type(rules["annotation_checked"]) is not bool or type(rules['auto_repair']) is not bool
            or type(rules['tighten_boxes']) is not bool):
        raise ValueError("Trạng thái xác nhận vùng phải là true/false/null.")
    indices = [b.get("ocr_index", i) for i, b in enumerate(boxes, 1)]
    if any(type(i) is not int or i < 1 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("Số thứ tự OCR nguồn không hợp lệ hoặc bị trùng.")
    for name in ("header_boxes", "exclude_boxes"):
        if (not isinstance(rules[name], list)
                or any(type(i) is not int or i not in indices for i in rules[name])):
            raise ValueError(f"{name}: có số box không tồn tại trong OCR gốc.")
    if set(rules["header_boxes"]) & set(rules["exclude_boxes"]):
        raise ValueError("Một box không thể vừa là trán bia vừa bị loại.")
    for name in ("header_regions", "exclude_regions"):
        if not isinstance(rules[name], list):
            raise ValueError(f"{name} phải là danh sách vùng.")
        for region in rules[name]:
            if (not isinstance(region, list) or len(region) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in region)
                    or not region[0] < region[2] or not region[1] < region[3]):
                raise ValueError(f"{name}: dùng vùng [x1,y1,x2,y2] hợp lệ.")
    header_marked = bool(rules["header_boxes"] or rules["header_regions"])
    if rules["header_present"] is False and header_marked:
        raise ValueError("Đã khai báo không có trán bia nhưng vẫn đánh dấu box/vùng trán bia.")
    if rules["header_present"] is True and not header_marked:
        raise ValueError("Cần chỉ ra box/vùng trán bia trước khi xác nhận có tiêu đề.")
    kept, excluded, seen = [], [], {}
    width, height = image_size
    for index, original in zip(indices, boxes):
        box = deepcopy(original)
        source_id = box.get("patch_id", f"ocr_{index:03d}")
        reason = None
        if not _valid_polygon(box.get("bbox")):
            reason = "invalid_polygon"
        elif rules['header_present'] is False and box.get('region_source') == 'horizontal_header_candidate':
            reason = 'header_candidate_rejected_by_no_header_rule'
        elif index in rules["exclude_boxes"]:
            reason = "explicit_exclusion"
        else:
            rect = _rect(box)
            if rect[2] <= 0 or rect[3] <= 0 or rect[0] >= width or rect[1] >= height:
                reason = "outside_image"
            elif any(_contained(rect, r) for r in rules["exclude_regions"]):
                reason = "excluded_annotation_region"
            signature = (tuple(tuple(p) for p in box["bbox"]), box.get("sino_nom_ocr", ""))
            if reason is None and signature in seen:
                reason = "exact_duplicate"
            if reason is None:
                seen[signature] = source_id
        if reason:
            excluded.append({"source_patch_id": source_id, "ocr_index": index, "reason": reason})
            continue
        box.update(source_patch_id=source_id, ocr_index=index, review_flags=[])
        automatic_header = (box.get('region_source') == 'horizontal_header_candidate'
                            and rules['header_present'] is not False)
        configured_header = index in rules["header_boxes"] or any(_contained(_rect(box), r) for r in rules["header_regions"])
        is_header = configured_header or automatic_header
        box["region"] = "header" if is_header else "body"
        box["is_header"] = is_header
        box["region_source"] = ('horizontal_header_candidate' if automatic_header and not configured_header else
                                "configured" if is_header or rules["header_present"] is False or header_marked else "body_default_unverified")
        if automatic_header and not configured_header:
            box['review_flags'].append('automatic_header_needs_review')
        rect = _rect(box)
        if rect[0] < 0 or rect[1] < 0 or rect[2] > width or rect[3] > height:
            box["review_flags"].append("partly_outside_image")
        if is_header and rect[2] - rect[0] > rect[3] - rect[1] and len(box.get("sino_nom_ocr", "")) > 1:
            box["review_flags"].append("header_text_direction_needs_review")
        kept.append(box)
    repairs = {'enabled': rules['auto_repair'], 'version': REPAIR_VERSION, 'changes': [],
               'applied_count': 0, 'proposal_count': 0, 'ocr_refresh_count': 0,
               'tail_review_count': 0, 'tail_reviews': [],
               'status': 'disabled' if not rules['auto_repair'] else 'not_run_no_image'}
    if rules['auto_repair'] and image is not None:
        excluded_indices = {b['ocr_index'] for b in excluded
                            if b['reason'] in {'explicit_exclusion', 'excluded_annotation_region'}}
        blockers = [b for index, b in zip(indices, boxes) if index in excluded_indices]
        repairs.update(repair_boxes(image, kept, blocked_boxes=blockers,
                                   blocked_regions=rules['exclude_regions'],
                                   refine_margins=rules['tighten_boxes']), status='completed')
        kept = repairs.pop('boxes')
    ordered = sort_boxes(kept)
    if len(ordered) > 999:
        raise ValueError("Quá 999 box; cần kiểm tra trước khi cấp số ba chữ số.")
    for index, box in enumerate(ordered, 1):
        box["patch_id"] = f"{box['image_id']}.{index:03d}"
        box["patch_path"] = None
        if 'tail_review' in box:
            box['tail_review']['reading_order'] = index
    source_orders = {source: b['reading_order'] for b in ordered
                     for source in b.get('source_ocr_indices', [b['ocr_index']])}
    for review in repairs['tail_reviews']:
        review['reading_order'] = source_orders.get(review['source_ocr_indices'][0])
    warnings = []
    if rules["header_present"] is None and not header_marked:
        warnings.append("header_not_confirmed")
    if not rules["annotation_checked"]:
        warnings.append("provenance_annotation_not_confirmed")
    if header_marked and not any(b['region'] == 'header' for b in ordered):
        warnings.append('configured_header_has_no_usable_box')
    warnings.extend(sorted({flag for box in ordered for flag in box['review_flags']}))
    if not ordered:
        warnings.append("no_usable_boxes")
    return {"boxes": ordered, "excluded": excluded, "rules": rules,
            'repair': repairs,
            "order_status": "provisional" if warnings else "configured",
            "review_required": True, "warnings": warnings}


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON phải là object: {path.name}")
    return value


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _preview(image, boxes, changes=()):
    with Image.open(image) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    scale = max(1.0, max(canvas.size) / 1024)
    font = ImageFont.load_default(size=round(10 * scale))
    for box in boxes:
        points = [(round(x), round(y)) for x, y in box["bbox"]]
        color = ('orange' if box['region'] == 'header' else
                 'magenta' if box.get('ocr_needs_refresh') else 'lime')
        draw.line(points + points[:1], fill=color, width=round(2 * scale))
        x, y = max(0, min(p[0] for p in points)), max(0, min(p[1] for p in points) - round(13 * scale))
        label = str(box["reading_order"])
        draw.rectangle(draw.textbbox((x, y), label, font=font), fill="black")
        draw.text((x, y), label, fill=color, font=font)
    for change in changes:
        if not change['applied']:
            points = [(round(x), round(y)) for x, y in change['bbox_after']]
            draw.line(points + points[:1], fill='yellow', width=max(1, round(scale)))
    stream = BytesIO()
    canvas.save(stream, format="PNG")
    return stream.getvalue()


def process_ocr_result(root, path, *, rules_override=None):
    root, path = Path(root).resolve(), Path(path).resolve()
    relative = path.relative_to(root).parts
    if (len(relative) != 7 or relative[:2] != ("data", "image_pipeline")
            or relative[4] != "02_ocr" or relative[6] not in {"ocr.json", "ocr_combined.json"}):
        raise ValueError("Chọn ocr.json trong thư mục 02_ocr của một ảnh.")
    book, stele, image_id = relative[2], relative[3], relative[5]
    if path != (ocr_folder(root, book, stele, image_id) / path.name).resolve():
        raise ValueError("Đường dẫn OCR không hợp lệ.")
    source_files = (combined_ocr_paths(path.parent) if path.name == 'ocr_combined.json' else
                    {key: path.parent / name for key, name in {
                        'raw': 'ocr_raw.json', 'boxes': 'ocr.json', 'preview': 'bbox_preview.png',
                        'complete': 'complete.json'}.items()})
    document, marker = _json(path), _json(source_files['complete'])
    if path.name == 'ocr_combined.json':
        from src.header_ocr import header_result_is_current
        if not header_result_is_current(root, book, stele, image_id):
            raise ValueError('Kết quả ghép trán bia chưa hợp lệ/cấu hình đã đổi; chạy lại luồng đầy đủ để cập nhật.')
    image = (prepare_folder(root, book, stele) / f"{image_id}.jpg").resolve()
    if (not image.is_relative_to(root) or document.get("image_id") != image_id or document.get("book_id") != book
            or (root / document["image_path"]).resolve() != image
            or document.get("image_sha256") != _file_hash(image)
            or marker.get("binding") != {k: document[k] for k in ("image_id", "image_sha256", "parameters")}):
        raise ValueError("Ảnh/JSON không khớp kết quả OCR đã lưu.")
    for key, filename in source_files.items():
        if key != 'complete' and marker["sha256"].get(key) != _file_hash(filename):
            raise ValueError(f"Kết quả OCR nguồn đã đổi/hỏng: {filename.name}")
    with Image.open(image) as decoded:
        size = decoded.size
    folder = filter_sort_folder(root, book, stele, image_id)
    rule_path = folder / "rules.json"
    rules = {**deepcopy(DEFAULT_RULES), **(_json(rule_path) if rule_path.exists() else {}),
             **(rules_override or {})}
    result = filter_and_sort(document["boxes"], size, rules, image=image)
    binding = {"algorithm_version": ALGORITHM_VERSION, "ocr_sha256": _file_hash(path),
               'ocr_source_file': path.name,
               'repair_version': REPAIR_VERSION,
               "image_sha256": document["image_sha256"], "rules_sha256": sha256(_encoded(rules)).hexdigest()}
    files = {"boxes": folder / "boxes.json", "preview": folder / "bbox_preview.png",
             "report": folder / "report.json", 'proposals_preview': folder / 'proposals_preview.png'}
    complete = folder / "complete.json"
    if complete.exists():
        old = _json(complete)
        for key, output in files.items():
            if key not in old.get('sha256', {}):
                if old.get('binding') == binding or output.exists():
                    raise ValueError('File kết quả không có checksum hợp lệ; giữ nguyên để kiểm tra.')
                continue  # Older algorithm versions did not generate a proposal preview.
            if not output.is_file() or _file_hash(output) != old.get("sha256", {}).get(key):
                raise ValueError("Kết quả lọc/sắp đã bị sửa; giữ nguyên để kiểm tra.")
        if old.get("binding") == binding:
            return _summary(root, files, result, "skipped_existing", book, stele, image_id)
        history = folder / "_history" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
        history.mkdir(parents=True, exist_ok=False)
        for output in [*files.values(), complete]:
            if not output.exists():
                continue
            target = (history / output.name).resolve()
            if not target.is_relative_to(root) or output.resolve().parent != folder.resolve():
                raise ValueError("Đường dẫn lịch sử không hợp lệ.")
            output.rename(target)
    folder.mkdir(parents=True, exist_ok=True)
    if not rule_path.exists() or _json(rule_path) != rules:
        _atomic_text(rule_path, _encoded(rules).decode("utf-8"))
    saved = {"schema_version": "box-order-2.0", "image_id": image_id, "book_id": book,
             "image_size": list(size), "coordinate_space": "prepared_image_pixels",
             "image_path": document["image_path"], "source_ocr_path": path.relative_to(root).as_posix(),
             "source_ocr_sha256": binding["ocr_sha256"], **result}
    report = {"input_boxes": len(document["boxes"]), "kept_boxes": len(result["boxes"]),
              "excluded_boxes": len(result["excluded"]), "order_status": result["order_status"],
              "warnings": result["warnings"], "excluded": result["excluded"],
              'repair': result['repair'], "rules": rules,
              "coordinates_changed": result['repair']['applied_count'] > 0,
              "ocr_text_changed": any(c['applied'] and c['action'] == 'merge' for c in result['repair']['changes']),
              'source_ocr_unchanged': True,
              'proposals_preview': files['proposals_preview'].relative_to(root).as_posix()}
    contents = {files["boxes"]: _encoded(saved), files["report"]: _encoded(report),
                files["preview"]: _preview(image, result["boxes"]),
                files['proposals_preview']: _preview(image, result['boxes'], result['repair']['changes'])}
    pending = {}
    for output, content in contents.items():
        if output.exists():
            if output.read_bytes() != content:
                raise FileExistsError(f"Output có sẵn không khớp: {output.name}")
        else:
            pending[output] = content
    _write_new_files(pending)
    if _file_hash(image) != binding['image_sha256'] or _file_hash(path) != binding['ocr_sha256']:
        raise ValueError('Nguồn thay đổi trong khi lọc/sắp; chưa ghi nhận hoàn thành.')
    _write_new_files({complete: _encoded({"binding": binding,
                                         "sha256": {k: _file_hash(v) for k, v in files.items()}})})
    return _summary(root, files, result, "success", book, stele, image_id)


def _summary(root, files, result, status, book, stele, image_id):
    return {"book_id": book, "stele_id": stele, "image_id": image_id, "status": status,
            "box_count": len(result["boxes"]), "excluded_count": len(result["excluded"]),
            "order_status": result["order_status"], "warnings": result["warnings"],
            'repair_applied': result['repair']['applied_count'],
            'repair_proposals': result['repair']['proposal_count'],
            'ocr_refresh_count': result['repair']['ocr_refresh_count'],
            'tail_review_count': result['repair']['tail_review_count'],
            'proposals_preview': files['proposals_preview'].relative_to(root).as_posix(),
            "sorted_json": files["boxes"].relative_to(root).as_posix(),
            "bbox_preview": files["preview"].relative_to(root).as_posix()}


def filter_result_is_current(root, book, stele, image_id, *, expected_source=None):
    """Lightweight queue eligibility check; explicit recheck verifies all hashes."""
    try:
        folder = filter_sort_folder(root, book, stele, image_id)
        marker = _json(folder / 'complete.json')
        rules = _json(folder / 'rules.json')
        binding = marker['binding']
        return (binding['algorithm_version'] == ALGORITHM_VERSION
                and (expected_source is None or binding.get('ocr_source_file') == expected_source)
                and binding['repair_version'] == REPAIR_VERSION
                and binding['rules_sha256'] == sha256(_encoded(rules)).hexdigest()
                and binding.get('ocr_source_file', 'ocr.json') in {'ocr.json', 'ocr_combined.json'}
                and binding['ocr_sha256'] == _file_hash(ocr_folder(root, book, stele, image_id)
                                                      / binding.get('ocr_source_file', 'ocr.json'))
                and all((folder / name).is_file() for name in ('boxes.json', 'report.json',
                                                              'bbox_preview.png', 'proposals_preview.png')))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def record_filter_result(root, row):
    """Called with the filter_sort process lock held by the full pipeline."""
    path = Path(root) / 'data/image_pipeline/filter_sort/progress.json'
    state = _json(path) if path.exists() else {'images': {}}
    if not isinstance(state.get('images'), dict):
        raise ValueError('File tiến độ lọc/sắp không hợp lệ; giữ lại để kiểm tra.')
    state['images'][f"{row['book_id']}/{row['image_id']}"] = row
    state['last_status'] = 'updated_by_full_pipeline'
    _atomic_text(path, _encoded(state).decode('utf-8'))


def _save_filter_reports(report, folder):
    report['summary'] = _counts(report['files'])
    provisional = sum(row.get('order_status') == 'provisional' for row in report['files'])
    report['report_warnings'] = ([f'{provisional} ảnh cần kiểm tra vùng, thứ tự hoặc box đã sửa/đề xuất.']
                                 if provisional else [])
    if not report['files']:
        report['report_warnings'].append('Chưa có kết quả OCR để lọc/sắp.')
    table = StringIO(newline='')
    writer = csv.writer(table)
    writer.writerow(['Sách', 'Mã bia', 'Ảnh', 'Trạng thái', 'Box giữ', 'Box loại',
                     'Lần tự sửa', 'Đề xuất cần xem', 'Box cần OCR lại',
                     'Thứ tự', 'Cảnh báo', 'Lỗi', 'JSON', 'Ảnh xem box', 'Ảnh đề xuất chưa áp dụng',
                     'Đuôi box cần đối chiếu'])
    for row in report['files']:
        writer.writerow([row.get('book_id'), row.get('stele_id'), row.get('image_id'),
                         row['status'], row.get('box_count'), row.get('excluded_count'),
                         row.get('repair_applied'), row.get('repair_proposals'), row.get('ocr_refresh_count'),
                         row.get('order_status'), ', '.join(row.get('warnings', [])),
                         row.get('reason'), row.get('sorted_json'), row.get('bbox_preview'), row.get('proposals_preview'),
                         row.get('tail_review_count')])
    count = report['summary']
    lines = ['# Báo cáo lọc / tự sửa / sắp box', '', f"Trạng thái: {report['status']}", '',
             f"Tổng {count['total']}: mới {count['success']}; đã có hợp lệ {count['skipped_existing']}; "
             f"lỗi {count['failed']}; bị ngắt {count['interrupted']}; chưa chạy {count['not_run']}.", '',
             f'{provisional} ảnh có thứ tự tạm. Kết quả cần kiểm tra trực quan trước khi crop.', '',
             'Không gọi API. OCR nguồn giữ nguyên; box đã sửa và liên kết nguồn lưu riêng. Chưa tạo patches/dataset.', '',
             f"Tự sửa {sum(r.get('repair_applied', 0) for r in report['files'])} lần; "
             f"đề xuất {sum(r.get('repair_proposals', 0) for r in report['files'])}; "
             f"box cần OCR lại {sum(r.get('ocr_refresh_count', 0) for r in report['files'])}.", '',
             f"Đuôi box cần đối chiếu văn bản: {sum(r.get('tail_review_count', 0) for r in report['files'])}.", '',
             'Chi tiết từng ảnh và nguyên nhân lỗi: report.csv; dữ liệu đầy đủ: report.json.']
    lines.extend(['', *report['report_warnings']])
    _atomic_text(folder / 'report.csv', table.getvalue(), encoding='utf-8-sig')
    _atomic_text(folder / 'summary.md', '\n'.join(lines) + '\n')
    _atomic_text(folder / 'report.json', _encoded(report).decode('utf-8'))


def run_filter_sort(root, *, selected_source=None, book_id=None, stele_id=None,
                    rules_override=None, progress=print):
    root = Path(root).resolve()
    paths = []
    if selected_source is not None:
        source = Path(selected_source)
        source = (root / source).resolve() if not source.is_absolute() else source.resolve()
        if (book_id not in SUPPORTED_BOOKS or not source.is_file()
                or source.parent != (root / "data/raw" / book_id).resolve()
                or infer_stele_id(source) != stele_id):
            raise ValueError("Mã sách/bia không khớp file nguồn.")
        for path in normalization_paths(root, book_id, stele_id):
            metadata = _json(path)
            if (root / metadata["source_path"]).resolve() == source:
                paths.append(preferred_ocr_path(ocr_folder(root, book_id, stele_id, metadata["image_id"])))
        paths = list(dict.fromkeys(paths))
        if not paths:
            raise ValueError("Ảnh chưa có kết quả chuẩn hóa/OCR.")
    else:
        if book_id is not None or stele_id is not None:
            raise ValueError('Mã sách/bia chỉ dùng khi chọn một ảnh.')
        if rules_override:
            raise ValueError("Chỉ áp dụng số box chỉ định cho một ảnh, không dùng chung cho All.")
        for book in sorted(SUPPORTED_BOOKS):
            paths.extend(preferred_ocr_path(path.parent) for path in
                         sorted((root / "data/image_pipeline" / book).glob("*/02_ocr/*/ocr.json")))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    report_dir = root / "data/image_pipeline/run_reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "stage": "box_filter_sort", "status": "running",
              "report_dir": str(report_dir),
              "files": [{"source_ocr": path.relative_to(root).as_posix(),
                         "book_id": path.parts[-5], "stele_id": path.parts[-4],
                         "image_id": path.parent.name, "status": "not_run"} for path in paths], "book_errors": [],
              "excluded": [], "report_warnings": []}
    with _single_run(root / "data/image_pipeline/filter_sort"):
        index_path = root / "data/image_pipeline/filter_sort/progress.json"
        state = _json(index_path) if index_path.exists() else {"images": {}}
        if not isinstance(state.get('images'), dict):
            raise ValueError('File tiến độ lọc/sắp không hợp lệ; giữ lại để kiểm tra.')
        try:
            for index, path in enumerate(paths, 1):
                try:
                    row = process_ocr_result(root, path, rules_override=rules_override)
                    state["images"][f"{row['book_id']}/{row['image_id']}"] = row
                    progress(f"[{index}/{len(paths)}] {row['image_id']}: {row['box_count']} box; {row['order_status']}.")
                except KeyboardInterrupt:
                    report['files'][index - 1]['status'] = 'interrupted'
                    raise
                except Exception as exc:
                    row = {**report['files'][index - 1],
                           "status": "failed", "reason": str(exc), "error_type": type(exc).__name__}
                    progress(f"[{index}/{len(paths)}] Lỗi: {path.parent.name}: {exc}")
                report["files"][index - 1] = row
                state['last_status'] = 'running'
                state['last_run_id'] = run_id
                _atomic_text(index_path, _encoded(state).decode("utf-8"))
                _save_filter_reports(report, report_dir)
            report["status"] = "completed"
        except KeyboardInterrupt:
            report["status"] = "interrupted"
        finally:
            state['last_status'] = report['status']
            _atomic_text(index_path, _encoded(state).decode('utf-8'))
            _save_filter_reports(report, report_dir)
    return report
