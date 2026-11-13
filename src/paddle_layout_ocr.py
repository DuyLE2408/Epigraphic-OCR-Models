"""Layout-guided local OCR: one recognition box per Title, column OCR in Content."""

from copy import deepcopy
import json
import math

from PIL import Image, ImageDraw

from src.local_layout import validate_layout
from src.paddle_ocr_trial import convert_result, write_json


def _polygon(rect):
    x0, y0, x1, y1 = rect
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _rect(polygon):
    return [min(p[0] for p in polygon), min(p[1] for p in polygon),
            max(p[0] for p in polygon), max(p[1] for p in polygon)]


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _blocked(polygon, regions):
    x0, y0, x1, y1 = _rect(polygon)
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    return any(r[0] <= center[0] <= r[2] and r[1] <= center[1] <= r[3] for r in regions)


def create_recognizer(settings):
    from paddleocr import TextRecognition
    args = {"model_name": settings["text_recognition_model_name"], "device": settings["device"],
            "cpu_threads": settings["cpu_threads"], "enable_mkldnn": settings["enable_mkldnn"]}
    if settings.get("text_recognition_model_dir"):
        args["model_dir"] = settings["text_recognition_model_dir"]
    return TextRecognition(**args)


def _save_raw(result, path):
    result.save_to_json(str(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw


def _crop(image, rect, folder, blocked, *, title=False):
    x0, y0, x1, y1 = rect
    dx, dy = ((x1 - x0) * .03, (y1 - y0) * .05) if title else (0, 0)
    crop_box = [max(0, math.floor(x0 - dx)), max(0, math.floor(y0 - dy)),
                min(image.width, math.ceil(x1 + dx)), min(image.height, math.ceil(y1 + dy))]
    crop = image.crop(crop_box)
    draw = ImageDraw.Draw(crop)
    masked = []
    for excluded in blocked:
        if _intersects(crop_box, excluded):
            # Working crop only. Exclude Title from body OCR and Notes from both.
            left, top = max(crop_box[0], math.floor(excluded[0])), max(crop_box[1], math.floor(excluded[1]))
            right, bottom = min(crop_box[2], math.ceil(excluded[2])), min(crop_box[3], math.ceil(excluded[3]))
            draw.rectangle((left - crop_box[0], top - crop_box[1], right - crop_box[0] - 1,
                            bottom - crop_box[1] - 1), fill="white")
            masked.append([left, top, right, bottom])
    path = folder / "input.png"
    crop.save(path)
    write_json(folder / "crop_metadata.json", {"crop_box_in_source_xyxy": crop_box,
                                               "layout_bbox_xyxy": rect, "masked_regions_in_source": masked,
                                               "resize_applied": False, "working_crop_only": True})
    return path, crop_box


def _translate(boxes, crop_box, region_id):
    output = deepcopy(boxes)
    for box in output:
        box["bbox_in_crop"] = box["bbox"]
        box["bbox"] = [[x + crop_box[0], y + crop_box[1]] for x, y in box["bbox"]]
        box["source_region"] = region_id
        box["bbox_source"] = "PaddleOCR polygon translated from Content crop"
        box["review_flags"] = [f for f in box["review_flags"] if f != "region_and_reading_order_unverified"]
        box["review_flags"].append("reading_order_unverified")
        if "ocr_index" in box:
            box["region"] = "body"
    return output


class LayoutRegionOCR:
    def __init__(self, layout, *, recognizer_factory=None, progress=print):
        self.layout = layout
        self.recognizer_factory = recognizer_factory or create_recognizer
        self.recognizer = None
        self.progress = progress

    def describe(self):
        return {"mode": "layout_regions", "layout": self.layout.describe(),
                "title": "one box per Title region; recognition only; text direction needs review",
                "content": "detection + recognition", "notes": "excluded; layout requires review"}

    def __call__(self, prepared, folder, image_id, image_size, ocr_model, settings):
        self.progress("  Layout: tìm Title / Content / Notes.")
        layout = self.layout(prepared, folder)
        validate_layout(layout, image_size)
        groups = {label: [p for p in layout["predictions"] if p["label"] == label]
                  for label in ("Title", "Content", "Notes")}
        if not groups["Content"]:
            raise ValueError("Layout không tìm được Content; chưa OCR để tránh đọc nhầm toàn ảnh.")
        notes = [p["bbox_xyxy"] for p in groups["Notes"]]
        titles = [p["bbox_xyxy"] for p in groups["Title"]]
        warnings = ["Layout and OCR require visual review; predictions are not ground truth.",
                    "Notes are excluded based on predicted layout; verify that genuine text was not excluded."]
        if not titles:
            warnings.append("title_not_found_by_layout")
        if len(titles) > 1:
            warnings.append("multiple_title_regions_need_review; not forced into one box")
        boxes, detected, excluded, regions = [], [], [], []
        region_root = folder / "regions"
        region_root.mkdir()
        with Image.open(prepared) as opened:
            image = opened.convert("RGB")
        for label in ("Title", "Content"):
            for index, item in enumerate(groups[label], 1):
                region_id = f"{label.lower()}_{index:03d}"
                target = region_root / region_id
                target.mkdir()
                rect = item["bbox_xyxy"]
                path, crop_box = _crop(image, rect, target, notes if label == "Title" else notes + titles,
                                       title=label == "Title")
                region_info = {"region_id": region_id, "label": label, "bbox_xyxy": rect,
                               "crop_box_in_source_xyxy": crop_box, "layout_confidence": item["confidence"]}
                regions.append(region_info)
                if label == "Title":
                    self.progress("  Title: nhận dạng nguyên tiêu đề, giữ một box.")
                    if self.recognizer is None:
                        self.recognizer = self.recognizer_factory(settings)
                    results = self.recognizer.predict(str(path))
                    if len(results) != 1:
                        raise ValueError("Expected one recognition result for a Title crop")
                    raw = _save_raw(results[0], target / "raw_result.json")
                    value = raw.get("res", raw)
                    text, score = value.get("rec_text"), value.get("rec_score")
                    if (not isinstance(text, str) or isinstance(score, bool)
                            or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
                        raise ValueError("Invalid Title recognition response; raw result preserved")
                    title_flags = ["visual_review_required", "header_text_direction_needs_review",
                                   "layout_boundary_needs_review"]
                    if not text.strip():
                        title_flags.append("title_without_recognized_text")
                    if any(_intersects(rect, n) for n in notes):
                        title_flags.append("title_overlaps_predicted_notes")
                    boxes.append({"image_id": image_id, "bbox": _polygon(rect), "region": "header",
                                  "sino_nom_ocr": text, "confidence": score, "reading_order": None,
                                  "reading_direction": "right_to_left", "sino_nom": None,
                                  "patch_id": None, "patch_path": None, "source_region": region_id,
                                  "layout_confidence": item["confidence"],
                                  "bbox_source": "DocLayout-YOLO Title, not an OCR character box",
                                  "review_flags": title_flags})
                    detected.append({"bbox": _polygon(rect), "confidence": item["confidence"],
                                     "source_region": region_id, "region": "header",
                                     "bbox_source": "DocLayout-YOLO Title", "review_flags": ["visual_review_required"]})
                else:
                    self.progress("  Content: phát hiện cột/dòng và nhận dạng; bỏ Title và Notes.")
                    results = ocr_model.predict(str(path))
                    if len(results) != 1:
                        raise ValueError("Expected one OCR result for a Content crop")
                    raw = _save_raw(results[0], target / "raw_result.json")
                    body = convert_result(raw, image_id=image_id,
                                          image_size=(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]))
                    write_json(target / "ocr_in_crop.json", body)
                    for kind, candidates, kept in (("recognition", body["boxes"], boxes),
                                                    ("detection", body["detected_boxes"], detected)):
                        for box in _translate(candidates, crop_box, region_id):
                            if _blocked(box["bbox"], notes + titles):
                                excluded.append({"kind": kind, "reason": "inside_title_or_notes", "box": box})
                            else:
                                kept.append(box)
        for i, box in enumerate(boxes, 1):
            box.update(ocr_index=i, box_id=f"{image_id}.box.{i:03d}")
        for i, box in enumerate(detected, 1):
            box["detection_index"] = i
        write_json(folder / "excluded_boxes.json", {"boxes": excluded})
        header_count = sum(b["region"] == "header" for b in boxes)
        header_recognized = sum(b["region"] == "header" and bool(b["sino_nom_ocr"].strip()) for b in boxes)
        self.progress(f"  Layout: {len(titles)} Title, {len(groups['Content'])} Content, {len(notes)} Notes bỏ qua.")
        return {"schema_version": "paddle-layout-ocr-trial-1.0", "image_id": image_id,
                "id_scope": "trial source filename; not final dataset ID",
                "coordinate_space": "prepared_image_pixels", "image_size": list(image_size),
                "box_count": len(boxes), "recognized_box_count": sum(bool(b["sino_nom_ocr"].strip()) for b in boxes),
                "detected_box_count": len(detected), "boxes": boxes, "detected_boxes": detected,
                "header_box_count": header_count, "header_recognized_count": header_recognized,
                "notes_region_count": len(notes), "regions": regions, "layout": layout,
                "review_required": True, "warnings": warnings,
                "note": "Title geometry comes from layout; body polygons come from PaddleOCR. OCR strings are preserved, not reversed."}
