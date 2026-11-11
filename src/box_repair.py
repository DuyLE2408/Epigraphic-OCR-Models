"""Local, conservative repair of seeded vertical columns; no OCR or network.

Geometry and pixel profiles are heuristics, not proof that a mark is a character.
Every automatic change retains its source boxes and requires visual review.
"""

from copy import deepcopy
import math
from statistics import median

from PIL import Image, ImageOps


REPAIR_VERSION = '1.2'


def _rect(box):
    return [min(p[0] for p in box['bbox']), min(p[1] for p in box['bbox']),
            max(p[0] for p in box['bbox']), max(p[1] for p in box['bbox'])]


def _polygon(rect):
    x1, y1, x2, y2 = rect
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _shape(box):
    points = sorted(box['bbox'], key=lambda p: (p[1], p[0]))
    tl, tr = sorted(points[:2])
    bl, br = sorted(points[2:])
    top, bottom = (tl[1] + tr[1]) / 2, (bl[1] + br[1]) / 2
    return tl[0], tr[0], bl[0], br[0], top, bottom


def _sides(box, y):
    left, right, lower_left, lower_right, top, bottom = _shape(box)
    scale = (y - top) / max(1, bottom - top)
    return left + scale * (lower_left - left), right + scale * (lower_right - right)


def _quad_at(box, top, bottom):
    left, right = _sides(box, top)
    lower_left, lower_right = _sides(box, bottom)
    return [[round(left, 3), top], [round(right, 3), top],
            [round(lower_right, 3), bottom], [round(lower_left, 3), bottom]]


def _width(box):
    left, right, lower_left, lower_right, _, _ = _shape(box)
    return (right - left + lower_right - lower_left) / 2


def _eligible(box):
    if box.get('region') == 'header' or box.get('is_header'):
        return False
    left, right, lower_left, lower_right, top, bottom = _shape(box)
    width, height = _width(box), bottom - top
    if width < 6 or height < 2 * width:
        return False
    drift = max(abs(lower_left - left), abs(lower_right - right))
    return drift / height <= 0.05 and min(right - left, lower_right - lower_left) >= 0.7 * width


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


def _obstructed(rect, boxes, ignored):
    for box in boxes:
        if id(box) in ignored:
            continue
        other = _rect(box)
        top, bottom = max(rect[1], other[1]), min(rect[3], other[3])
        if bottom <= top:
            continue
        overlaps = []
        for y in (top, (top + bottom) / 2, bottom):
            left, right = _sides(box, y)
            overlaps.append(max(0, min(rect[2], right) - max(rect[0], left))
                            / max(1, min(rect[2] - rect[0], right - left)))
        if sum(overlaps) / len(overlaps) > 0.3:
            return True
    return False


def _otsu(histogram):
    total = sum(histogram)
    overall = sum(i * count for i, count in enumerate(histogram))
    left_count = left_sum = 0
    best, threshold = -1, 0
    for value, count in enumerate(histogram):
        left_count += count
        left_sum += value * count
        right_count = total - left_count
        if not left_count or not right_count:
            continue
        variance = left_count * right_count * (left_sum / left_count - (overall - left_sum) / right_count) ** 2
        if variance > best:
            best, threshold = variance, value
    return threshold if best > 0 else None


def _profile(gray, rect, box=None, *, details=False, inset_ratio=0.1):
    """Infer minority ink polarity inside the seed; measure rows in its strip."""
    width, height = gray.size
    x1, y1, x2, y2 = rect
    inset = max(0, round((_width(box) if box is not None else x2 - x1) * inset_ratio))
    left, right = max(0, math.ceil(x1) + inset), min(width, math.floor(x2) - inset)
    top, bottom = max(0, math.ceil(y1)), min(height, math.floor(y2))
    if right - left < 3 or bottom <= top:
        return None
    pixels = gray.tobytes()
    rows = []
    for y in range(height):
        a, b = _sides(box, y) if box is not None else (x1, x2)
        a, b = max(0, math.ceil(a) + inset), min(width, math.floor(b) - inset)
        rows.append(pixels[y * width + a:y * width + b] if b > a else b'')
    histogram = [0] * 256
    for row in rows[top:bottom]:
        for value in row:
            histogram[value] += 1
    threshold = _otsu(histogram)
    if threshold is None:
        return None
    dark = sum(histogram[:threshold + 1])
    total = sum(histogram)
    light_ink = dark > total / 2
    ink_count = total - dark if light_ink else dark
    if not 0.01 <= ink_count / total <= 0.45:
        return None
    light = total - dark
    dark_mean = sum(i * histogram[i] for i in range(threshold + 1)) / max(1, dark)
    light_mean = sum(i * histogram[i] for i in range(threshold + 1, 256)) / max(1, light)
    if light_mean - dark_mean < 32:
        return None
    lookup = [int(v > threshold) if light_ink else int(v <= threshold) for v in range(256)]
    span = max(1, round(sum(len(row) for row in rows[top:bottom]) / (bottom - top)))
    binary = []
    unit = max(6, round(_width(box)) if box is not None else span)
    local_lookups = {}
    for y, row in enumerate(rows):
        table = lookup
        if y < top or y >= bottom:
            block = y // unit
            if block not in local_lookups:
                local_hist = [0] * 256
                for nearby in rows[max(0, block * unit - unit):min(height, (block + 2) * unit)]:
                    for value in nearby:
                        local_hist[value] += 1
                local_threshold = _otsu(local_hist)
                local_table = lookup
                if local_threshold is not None:
                    total_local = sum(local_hist)
                    dark_local = sum(local_hist[:local_threshold + 1])
                    light_local = total_local - dark_local
                    dark_avg = sum(i * local_hist[i] for i in range(local_threshold + 1)) / max(1, dark_local)
                    light_avg = sum(i * local_hist[i] for i in range(local_threshold + 1, 256)) / max(1, light_local)
                    fraction = (light_local if light_ink else dark_local) / max(1, total_local)
                    if 0.02 <= fraction <= 0.45 and light_avg - dark_avg >= 24:
                        local_table = [int(v > local_threshold) if light_ink else int(v <= local_threshold) for v in range(256)]
                local_lookups[block] = local_table
            table = local_lookups[block]
        binary.append([table[value] for value in row])
    counts = list(map(sum, binary))
    active = [max(2, span * 0.08) <= count <= span * 0.75 for count in counts]
    activity = sum(active[top:bottom]) / (bottom - top)
    if not 0.15 <= activity <= 0.92:
        return None
    return (active, counts, span, binary) if details else (active, counts, span)


def _cell_evidence(rows, start, end, unit):
    cells = []
    for y in range(max(0, math.floor(start)), min(len(rows), math.ceil(end)), unit):
        data = rows[y:min(math.ceil(end), y + unit)]
        area = sum(map(len, data))
        if len(data) < unit * 0.6 or not area:
            continue
        width = max(map(len, data))
        transitions = sum(sum(a != b for a, b in zip(row, row[1:])) for row in data) / area
        coverage = sum(sum(row[x] for row in data if x < len(row)) >= max(1, len(data) * 0.05)
                       for x in range(width)) / max(1, width)
        cells.append({'density': sum(map(sum, data)) / area, 'transitions': transitions,
                      'coverage': coverage, 'activity': sum(sum(row) >= 2 for row in data) / len(data)})
    return cells


def _extension_evidence(rows, rect, target, direction, width):
    unit = max(6, round(width))
    anchor = _cell_evidence(rows, rect[1], rect[3], unit)
    anchor = [c for c in anchor if 0.05 <= c['density'] <= 0.5 and c['coverage'] >= 0.4]
    extra = _cell_evidence(rows, min(rect[1], target) if direction < 0 else rect[3],
                           rect[1] if direction < 0 else max(rect[3], target), unit)
    if not anchor or not extra:
        return False, 0
    baseline = {key: median(c[key] for c in anchor) for key in ('density', 'transitions')}
    supported = [c for c in extra if max(0.05, baseline['density'] * 0.45) <= c['density'] <= 0.5
                 and c['coverage'] >= 0.4 and c['activity'] >= 0.35
                 and c['transitions'] >= max(0.04, baseline['transitions'] * 0.45)]
    consistent = (supported and median(c['density'] for c in supported) >= baseline['density'] * 0.7
                  and median(c['transitions'] for c in supported) >= baseline['transitions'] * 0.65)
    return bool(consistent and len(supported) >= 2 and len(supported) / len(extra) >= 0.75), len(supported)


def _tighten_horizontal(gray, box):
    """Trim only low-mass side margins; do not decide which glyphs are genuine."""
    rect, width = _rect(box), _width(box)
    profile = _profile(gray, rect, box, details=True, inset_ratio=0)
    if profile is None:
        return None
    histogram = [0] * 64
    for row in profile[3][max(0, math.ceil(rect[1])):min(gray.height, math.floor(rect[3]))]:
        for x, value in enumerate(row):
            if value:
                histogram[min(63, int((x + 0.5) / len(row) * 64))] += 1
    total = sum(histogram)
    if total < width:
        return None
    edges = []
    for sequence in (list(enumerate(histogram)), list(reversed(list(enumerate(histogram))))):
        cumulative = 0
        for index, mass in sequence:
            cumulative += mass
            if cumulative > total * 0.02:
                edges.append(index)
                break
    left = min(0.2, max(0, edges[0] / 64 - 1.5 / width))
    right = max(0.8, min(1, (edges[1] + 1) / 64 + 1.5 / width))
    if max(left, 1 - right) * width < 1 or right - left < 0.65:
        return None
    a, b, c, d, top, bottom = _shape(box)
    polygon = [[round(a + left * (b - a), 3), top], [round(a + right * (b - a), 3), top],
               [round(c + right * (d - c), 3), bottom], [round(c + left * (d - c), 3), bottom]]
    removed = sum(mass for index, mass in enumerate(histogram)
                  if (index + 0.5) / 64 < left or (index + 0.5) / 64 > right)
    if removed / total > 0.01:
        return None
    return polygon, removed / total


def _tail_review(gray, box):
    rect, unit = _rect(box), max(6, round(_width(box)))
    profile = _profile(gray, rect, box, details=True)
    if profile is None:
        return None
    rows = profile[3]
    anchor = _cell_evidence(rows, rect[1], rect[3] - 2 * unit, unit)
    anchor = [c for c in anchor if c['density'] >= 0.05 and c['coverage'] >= 0.4]
    tail = _cell_evidence(rows, max(rect[1], rect[3] - 2 * unit), rect[3], unit)
    if not anchor or not tail:
        return None
    baseline = median(c['density'] for c in anchor)
    weak = [i for i, c in enumerate(tail) if c['density'] < max(0.05, baseline * 0.45)
            or c['coverage'] < 0.4 or c['activity'] < 0.3]
    if not weak:
        return None
    return {'source_ocr_indices': box.get('source_ocr_indices', [box['ocr_index']]),
            'bbox': _quad_at(box, max(rect[1], rect[3] - 2 * unit), rect[3]),
            'reason': 'weak_tail_marks_not_verified_as_characters',
            'weak_cells': weak, 'action': 'review_only', 'requires_text_alignment': True}


def _peer_boundary_support(box, target, direction, boxes):
    width = _width(box)
    center = sum(_sides(box, target)) / 2
    peers = 0
    for other in boxes:
        if other is box or not _eligible(other):
            continue
        other_width = _width(other)
        distance = abs(sum(_sides(other, target)) / 2 - center)
        end = _shape(other)[4 if direction < 0 else 5]
        if (0.65 * min(width, other_width) <= distance <= 4 * max(width, other_width)
                and abs(end - target) <= 1.5 * width):
            peers += 1
    return peers >= 2


def _sources(box):
    return deepcopy(box.get('source_boxes') or [
        {key: deepcopy(box.get(key)) for key in ('ocr_index', 'patch_id', 'bbox', 'sino_nom_ocr', 'confidence')}
    ])


def _mark_changed(box, sources, flag):
    box['source_boxes'] = sources
    box['source_ocr_indices'] = [b['ocr_index'] for b in sources]
    box['source_patch_ids'] = [b['patch_id'] for b in sources]
    box['ocr_needs_refresh'] = True
    box['sino_nom'] = None
    box['patch_path'] = None
    box['review_flags'] = sorted(set(box.get('review_flags', [])) | {flag, 'ocr_needs_refresh'})


def _merge_fragments(boxes, gray, blockers):
    changes = []
    # Recompute candidates after each merge so a chain cannot merge around a blocker.
    while True:
        candidates = []
        for first in boxes:
            if not _eligible(first):
                continue
            a = _rect(first)
            for second in boxes:
                if first is second or not _eligible(second):
                    continue
                b = _rect(second)
                aw, bw = _width(first), _width(second)
                gap = b[1] - a[3]
                al, ar = _sides(first, (a[3] + b[1]) / 2)
                bl, br = _sides(second, (a[3] + b[1]) / 2)
                if not (0 <= gap <= 1.25 * min(aw, bw)
                        and max(aw, bw) <= 1.25 * min(aw, bw)
                        and abs((al + ar - bl - br) / 2) <= 0.3 * min(aw, bw)
                        and max(0, min(ar, br) - max(al, bl)) >= 0.7 * min(aw, bw)):
                    continue
                union = [min(a[0], b[0]), a[1], max(a[2], b[2]), b[3]]
                bridge = [min(al, bl), a[3], max(ar, br), b[1]]
                if _obstructed(bridge, boxes + blockers, {id(first), id(second)}):
                    continue
                profile = _profile(gray, a, first)
                if profile is None or _profile(gray, b, second) is None:
                    continue
                # A broad solid stroke in the gap may be a separator/frame.
                _, counts, span = profile
                if any(count >= 0.85 * span for count in counts[math.ceil(a[3]):math.floor(b[1])]):
                    continue
                candidates.append((gap, first['ocr_index'], second['ocr_index'], first, second, union))
        if not candidates:
            return boxes, changes
        _, _, _, first, second, union = min(candidates, key=lambda item: item[:3])
        sources = _sources(first) + _sources(second)
        merged = deepcopy(first)
        merged['bbox'] = _polygon(union)
        merged['sino_nom_ocr'] = ''.join(b.get('sino_nom_ocr') or '' for b in sources)
        merged['ocr_text_status'] = 'joined_fragments_unverified'
        scores = [b.get('confidence') for b in sources]
        merged['confidence'] = min(scores) if all(type(s) in (int, float) for s in scores) else None
        _mark_changed(merged, sources, 'auto_merged_fragments')
        changes.append({'action': 'merge', 'applied': True, 'source_ocr_indices': merged['source_ocr_indices'],
                        'bbox_before': [first['bbox'], second['bbox']], 'bbox_after': merged['bbox'],
                        'reason': 'aligned_vertical_fragments_with_short_gap'})
        boxes = [b for b in boxes if b is not first and b is not second] + [merged]


def _scan(active, start, direction, width, barriers):
    """Follow nearby ink; stop at a blank gap, image edge or bounded search."""
    gap_limit, search = max(2, round(width * 0.85)), max(4, round(width * 6))
    ink, blank = [], 0
    exhausted = True
    for step in range(search):
        y = start + direction * step
        if y < 0 or y >= len(active):
            exhausted = False
            break
        if barriers[y]:
            exhausted = False
            break
        if active[y]:
            ink.append(y)
            blank = 0
        else:
            blank += 1
            allowed_gap = gap_limit if ink else max(gap_limit, round(width * 1.5))
            if blank > allowed_gap:
                exhausted = False
                break
    if len(ink) < max(3, round(width * 0.3)) or max(ink) - min(ink) < width * 0.45:
        return None
    edge = min(ink) - 1 if direction < 0 else max(ink) + 2
    return edge, exhausted


def repair_boxes(image, boxes, *, blocked_boxes=(), blocked_regions=(), refine_margins=True):
    """Return copies with applied repairs and explicit un-applied proposals.

Input boxes must already have valid finite convex polygons and region labels.
No image file is modified. Long/blocked/ambiguous extensions remain proposals.
"""
    boxes = deepcopy(boxes)
    if isinstance(image, Image.Image):
        gray = ImageOps.grayscale(image)
    else:
        with Image.open(image) as decoded:
            gray = ImageOps.grayscale(decoded)
    original = deepcopy(boxes)
    blockers = list(blocked_boxes) + [{'bbox': _polygon(rect)} for rect in blocked_regions]
    boxes, changes = _merge_fragments(boxes, gray, blockers)
    body = [b for b in original if _eligible(b)]
    envelope = (min((_rect(b)[1] for b in body), default=0),
                max((_rect(b)[3] for b in body), default=gray.height))
    unassessed = []
    for box in boxes:
        if not _eligible(box):
            continue
        rect = _rect(box)
        profile = _profile(gray, rect, box, details=True)
        if profile is None:
            unassessed.append({'source_ocr_indices': box.get('source_ocr_indices', [box['ocr_index']]),
                               'reason': 'insufficient_pixel_evidence'})
            continue
        active, counts, span, binary = profile
        barriers = [count >= 0.85 * span for count in counts]
        width = _width(box)
        for direction in (-1, 1):
            edge = 1 if direction < 0 else 3
            start = math.ceil(rect[1]) - 1 if direction < 0 else math.floor(rect[3])
            found = _scan(active, start, direction, width, barriers)
            if found is None:
                continue
            target, exhausted = found
            target = max(0, min(gray.height, target))
            candidate = list(rect)
            candidate[edge] = target
            polygon = _quad_at(box, candidate[1], candidate[3])
            added_polygon = _quad_at(box, min(rect[edge], target), max(rect[edge], target))
            added = [min(p[0] for p in added_polygon), min(rect[edge], target),
                     max(p[0] for p in added_polygon), max(rect[edge], target)]
            distance = (rect[edge] - target) * -direction
            if distance <= 1:
                continue
            reason = 'nearby_ink_continuation'
            applied = True
            supported, supported_cells = _extension_evidence(binary, rect, target, direction, width)
            peer_boundary = _peer_boundary_support(box, target, direction, boxes)
            supported = supported or (supported_cells >= 1 and peer_boundary and distance >= 1.5 * width)
            if not supported:
                reason, applied = 'weak_character_continuation', False
            elif exhausted or (distance > width * 2.5 and not (peer_boundary and distance <= width * 5)):
                reason, applied = 'long_extension_needs_review', False
            elif target < envelope[0] - width or target > envelope[1] + width:
                reason, applied = 'outside_seeded_body_envelope', False
            elif any(x < 0 or x > gray.width for x, _ in polygon):
                reason, applied = 'outside_image', False
            elif _obstructed(added, boxes + blockers, {id(box)}):
                reason, applied = 'overlaps_another_box', False
            change = {'action': 'extend_top' if direction < 0 else 'extend_bottom',
                      'applied': applied, 'source_ocr_indices': box.get('source_ocr_indices', [box['ocr_index']]),
                      'bbox_before': deepcopy(box['bbox']), 'bbox_after': polygon, 'reason': reason,
                      'supported_character_cells': supported_cells}
            changes.append(change)
            if applied:
                sources = _sources(box)
                box['bbox'] = polygon
                box['ocr_text_status'] = 'partial_after_box_extension'
                _mark_changed(box, sources, 'auto_extended_box')
                rect = candidate
            else:
                box['review_flags'] = sorted(set(box.get('review_flags', [])) | {'box_extension_needs_review'})
    tail_reviews = []
    for box in boxes:
        if not _eligible(box):
            continue
        tight = _tighten_horizontal(gray, box) if refine_margins else None
        if tight is not None:
            polygon, removed_fraction = tight
            sources, before = _sources(box), deepcopy(box['bbox'])
            box['bbox'] = polygon
            _mark_changed(box, sources, 'auto_tightened_box')
            box.setdefault('ocr_text_status', 'unchanged_after_margin_tightening')
            changes.append({'action': 'tighten_sides', 'applied': True,
                            'source_ocr_indices': box['source_ocr_indices'],
                            'bbox_before': before, 'bbox_after': polygon,
                            'reason': 'low_ink_side_margins', 'removed_foreground_fraction': removed_fraction})
        review = _tail_review(gray, box)
        if review is not None:
            tail_reviews.append(review)
            box['tail_review'] = review
            box['review_flags'] = sorted(set(box.get('review_flags', [])) | {'tail_marks_need_text_alignment'})
    return {'boxes': boxes, 'version': REPAIR_VERSION, 'changes': changes,
            'applied_count': sum(c['applied'] for c in changes),
            'proposal_count': sum(not c['applied'] for c in changes),
            'ocr_refresh_count': sum(bool(b.get('ocr_needs_refresh')) for b in boxes),
            'unassessed': unassessed,
            'tail_reviews': tail_reviews,
            'tail_review_count': len(tail_reviews),
            'review_required': True}
