"""Isolated PP-OCRv6 experiment, optionally guided by a local layout processor.

PaddleOCR is imported lazily so data conversion can be tested without models.
The original detection results and recognition-filtered boxes remain separate.
"""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from src.image_ocr import load_source_image, normalize_image


MODEL_VERSION = "PP-OCRv6"
SUPPORTED_INPUTS = {".jpg", ".jpeg", ".png", ".pdf"}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def model_settings(size="medium", cpu_threads=4, rec_score_thresh=0.0,
                   det_max_side=None, det_model_dir=None, rec_model_dir=None, *, device="cpu"):
    if size not in {"tiny", "small", "medium"}:
        raise ValueError("Model size must be tiny, small or medium")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    if isinstance(rec_score_thresh, bool) or not 0 <= rec_score_thresh <= 1:
        raise ValueError("rec_score_thresh must be between 0 and 1")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|gpu:[0-9]+", device):
        raise ValueError("OCR device must be cpu or gpu:N")
    settings = {
        "ocr_version": MODEL_VERSION, "lang": "chinese_cht", "device": device,
        "text_detection_model_name": f"{MODEL_VERSION}_{size}_det",
        "text_recognition_model_name": f"{MODEL_VERSION}_{size}_rec",
        "use_doc_orientation_classify": False, "use_doc_unwarping": False,
        "use_textline_orientation": False, "enable_mkldnn": False,
        "cpu_threads": cpu_threads, "text_rec_score_thresh": rec_score_thresh,
    }
    if det_max_side is not None:
        if type(det_max_side) is not int or det_max_side < 32:
            raise ValueError("det_max_side must be an integer of at least 32")
        settings.update(text_det_limit_side_len=det_max_side, text_det_limit_type="max")
    for key, directory in (("text_detection_model_dir", det_model_dir),
                           ("text_recognition_model_dir", rec_model_dir)):
        if directory is not None:
            path = Path(directory).resolve()
            if not path.is_dir():
                raise ValueError(f"Model directory does not exist: {path}")
            settings[key] = str(path)
    return settings


def create_model(settings):
    # This disables model-host probing, not TLS verification. Missing weights may
    # still download from the official model host; images stay on this machine.
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise RuntimeError(
            "Chưa có PaddleOCR trong Python đang chọn. "
            "Xem docs/paddle_ocr_trial.md và dùng .venv-ocr."
        ) from None
    return PaddleOCR(**settings)


def _number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _polygon(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(not isinstance(p, list) or len(p) != 2
                   or any(not _number(v) for v in p) for p in value)):
        raise ValueError("OCR polygon must have four finite 2D points")
    area = abs(sum(value[i][0] * value[(i + 1) % 4][1]
                   - value[(i + 1) % 4][0] * value[i][1] for i in range(4))) / 2
    if not area:
        raise ValueError("OCR polygon has zero area")
    return value


def convert_result(raw, *, image_id, image_size):
    """Never zip rec_texts with dt_polys: detection and recognition can differ."""
    if not isinstance(raw, dict):
        raise ValueError("PaddleOCR did not save a JSON object")
    data = raw.get("res", raw)
    if not isinstance(data, dict):
        raise ValueError("Invalid PaddleOCR result envelope")
    detection = data.get("dt_polys")
    recognized = data.get("rec_polys")
    texts, scores = data.get("rec_texts"), data.get("rec_scores")
    if (not isinstance(detection, list) or not isinstance(recognized, list)
            or not isinstance(texts, list) or not isinstance(scores, list)
            or len(recognized) != len(texts) or len(texts) != len(scores)
            or any(not isinstance(t, str) for t in texts)
            or any(not _number(s) or not 0 <= s <= 1 for s in scores)):
        raise ValueError("Recognition boxes, texts and scores do not match; raw JSON preserved")
    detection_scores = data.get("dt_scores")
    if detection_scores is not None and (
            not isinstance(detection_scores, list) or len(detection_scores) != len(detection)
            or any(not _number(s) or not 0 <= s <= 1 for s in detection_scores)):
        raise ValueError("Detection scores do not match detection boxes")
    width, height = image_size
    def flags(polygon):
        result = ["visual_review_required", "region_and_reading_order_unverified"]
        if any(not 0 <= p[0] <= width or not 0 <= p[1] <= height for p in polygon):
            result.append("coordinates_outside_prepared_image")
        return result
    boxes = []
    for i, (polygon, text, score) in enumerate(zip(recognized, texts, scores), 1):
        polygon = _polygon(polygon)
        boxes.append({
            "image_id": image_id, "box_id": f"{image_id}.box.{i:03d}",
            "ocr_index": i, "bbox": polygon, "sino_nom_ocr": text,
            "confidence": score, "reading_order": None, "region": "unclassified",
            "patch_id": None, "patch_path": None, "sino_nom": None,
            "review_flags": flags(polygon),
        })
    detected_boxes = []
    for i, polygon in enumerate(detection, 1):
        polygon = _polygon(polygon)
        detected_boxes.append({"detection_index": i, "bbox": polygon,
                               "confidence": detection_scores[i - 1] if detection_scores is not None else None,
                               "review_flags": flags(polygon)})
    return {
        "schema_version": "paddle-ocr-trial-1.0", "image_id": image_id,
        "id_scope": "trial source filename; not final dataset ID",
        "coordinate_space": "prepared_image_pixels", "image_size": list(image_size),
        "bbox_source": "PaddleOCR rec_polys / dt_polys, preserved without box repair",
        "detected_box_count": len(detected_boxes), "recognized_box_count": len(boxes),
        "boxes": boxes, "detected_boxes": detected_boxes, "review_required": True,
        "detection_parameters": data.get("text_det_params"),
        "recognition_score_threshold": data.get("text_rec_score_thresh"),
        "warnings": ["No Title/Content/Notes classification or annotation removal has been performed.",
                     "OCR predictions are not ground truth; reading order is not confirmed."],
    }


def _preview(image_path, boxes, *, detection=False):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = max(1, max(image.size) / 1500)
    font = ImageFont.load_default(size=round(13 * scale))
    for box in boxes:
        points = [(round(x), round(y)) for x, y in box["bbox"]]
        header = box.get("region") == "header"
        draw.line(points + points[:1], fill="orange" if header else "lime", width=max(2, round(2 * scale)))
        index = box["detection_index"] if detection else box["ocr_index"]
        label = "Title" if header else f"D{index:03d}" if detection else f"{index:03d}"
        x = max(0, min(image.width - round(40 * scale), min(p[0] for p in points)))
        y = max(0, min(image.height - round(18 * scale), min(p[1] for p in points) - round(17 * scale)))
        draw.rectangle(draw.textbbox((x, y), label, font=font), fill="black")
        draw.text((x, y), label, font=font, fill="yellow")
    return image


def _prepare(source, folder, render_dpi):
    if source.suffix.lower() == ".png":
        from PIL import ImageOps
        with Image.open(source) as original:
            if original.format != "PNG":
                raise ValueError("PNG extension does not match image content")
            orientation = original.getexif().get(274, 1)
            image = ImageOps.exif_transpose(original).copy()
            inspection = {"source_type": "png", "encoded_image_size": list(original.size),
                          "exif_orientation": orientation,
                          "exif_orientation_applied": orientation in {2, 3, 4, 5, 6, 7, 8}}
    else:
        # Reuse read-only loading from the existing image pipeline. This does not
        # invoke normalization output paths, page-reference updates or school OCR.
        image, inspection = load_source_image(source, render_dpi=render_dpi)
    rgb = normalize_image(image, max_dim=None)
    can_copy = (source.suffix.lower() in {".jpg", ".jpeg"} and image.mode == "RGB"
                and not inspection.get("exif_orientation_applied"))
    prepared = folder / ("input.jpg" if can_copy else "input.png")
    if can_copy:
        with prepared.open("xb") as stream, source.open("rb") as original:
            shutil.copyfileobj(original, stream)
        method = "copy_original_jpeg"
    else:
        with prepared.open("xb") as stream:
            rgb.save(stream, format="PNG")
        method = "lossless_rgb_png"
    return prepared, {"image_size": list(rgb.size), "source_inspection": inspection,
                      "preparation": method, "external_resize_applied": False,
                      "note": "Detector may internally resize; output coordinates refer to this prepared image."}


def run_trial(sources, output_root, settings, *, render_dpi=300, model_factory=None,
              region_processor=None, progress=print):
    """Sequential experiment, fresh directory each run; preserve existing outputs.

    model_factory is injectable for tests. No model is imported by module import.
    """
    inputs = [Path(p).resolve() for p in sources]
    if not inputs:
        raise ValueError("Chưa chọn ảnh để thử")
    if type(render_dpi) is not int or render_dpi < 72:
        raise ValueError("render_dpi must be an integer of at least 72")
    output_root = Path(output_root).resolve()
    for source in inputs:
        if not source.is_file() or source.suffix.lower() not in SUPPORTED_INPUTS:
            raise ValueError(f"Ảnh/PDF không hợp lệ: {source}")
        if source.is_relative_to(output_root):
            raise ValueError("Không chọn input bên trong thư mục kết quả thử nghiệm")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ("paddleocr", "paddlepaddle", "paddlepaddle-gpu", "paddlex", "Pillow", "PyMuPDF"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    manifest = {"run_id": run_id, "model_settings": settings, "packages": versions,
                "render_dpi": render_dpi, "sources": [str(s) for s in inputs],
                "scope": "Local layout + PaddleOCR" if region_processor else "PaddleOCR only",
                "excluded_stages": "school API, box repair, patches and dataset export",
                "region_processing": region_processor.describe() if region_processor else None}
    write_json(output / "manifest.json", manifest)
    progress(f"Kết quả thử riêng: {output}")
    rows = []
    report = {"run_id": run_id, "run_dir": str(output), "model_settings": settings,
              "status": "running", "files": rows, "review_required": True}
    model = None
    model_error = None
    interrupted = False
    try:
        progress("Đang nạp PP-OCRv6 local; lần đầu có thể tải model chính thức.")
        try:
            model = (model_factory or create_model)(settings)
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"
        for i, source in enumerate(inputs, 1):
            slug = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:70] or "image"
            folder = output / f"{i:03d}_{slug}"
            folder.mkdir()
            row = {"source": str(source), "output_dir": str(folder), "status": "running"}
            rows.append(row)
            progress(f"[{i}/{len(inputs)}] {source.name}")
            started = time.perf_counter()
            before = None
            try:
                if model_error:
                    raise RuntimeError(model_error)
                before = file_hash(source)
                row["source_sha256"] = before
                prepared, preparation = _prepare(source, folder, render_dpi)
                prepared_hash = file_hash(prepared)
                write_json(folder / "input_metadata.json", {
                    **preparation, "source": str(source), "source_sha256": before,
                    "prepared_sha256": prepared_hash,
                })
                if region_processor:
                    document = region_processor(prepared, folder, source.stem,
                                                preparation["image_size"], model, settings)
                else:
                    results = model.predict(str(prepared))
                    if len(results) != 1:
                        raise ValueError(f"Expected one OCR result, received {len(results)}")
                    raw_path = folder / "raw_result.json"
                    results[0].save_to_json(str(raw_path))
                    with raw_path.open(encoding="utf-8") as stream:
                        raw = json.load(stream)
                    document = convert_result(raw, image_id=source.stem,
                                              image_size=preparation["image_size"])
                if file_hash(source) != before or file_hash(prepared) != prepared_hash:
                    raise ValueError("Input changed during OCR; result is not accepted")
                document.update(source=str(source), source_sha256=before, source_unchanged=True,
                                prepared_image=prepared.name, prepared_sha256=prepared_hash,
                                model_settings=settings, elapsed_seconds=round(time.perf_counter() - started, 3))
                write_json(folder / "ocr.json", document)
                _preview(prepared, document["boxes"]).save(folder / "bbox_preview.png")
                _preview(prepared, document["detected_boxes"], detection=True).save(folder / "detection_preview.png")
                row.update(status="saved_for_review" if document["recognized_box_count"] else "no_recognized_text",
                           detected_boxes=document["detected_box_count"], recognized_boxes=document["recognized_box_count"],
                           image_size=preparation["image_size"], source_unchanged=True,
                           ocr_json=str(folder / "ocr.json"), preview=str(folder / "bbox_preview.png"))
                if region_processor:
                    row.update(header_box_count=document["header_box_count"],
                               header_recognized_count=document["header_recognized_count"],
                               notes_region_count=document["notes_region_count"],
                               layout_preview=str(folder / "layout/layout_preview.png"),
                               review_warnings=document["warnings"])
                    progress(f"  Giữ {len(document['boxes'])} box, {row['recognized_boxes']} box có chữ; "
                             f"{row['header_box_count']} box tiêu đề. Cần xem preview.")
                else:
                    progress(f"  Detection: {row['detected_boxes']} box; recognition: {row['recognized_boxes']} box. Cần xem preview.")
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                if before is not None:
                    row["source_unchanged"] = source.is_file() and file_hash(source) == before
                write_json(folder / "error.json", row)
                progress(f"  Lỗi: {type(exc).__name__}: {exc}")
            finally:
                row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    except KeyboardInterrupt:
        interrupted = True
        if rows and rows[-1]["status"] == "running":
            rows[-1]["status"] = "interrupted"
        progress("Đã dừng; các file kết quả đã lưu được giữ nguyên.")
    finally:
        statuses = [row["status"] for row in rows]
        report.update(status="interrupted" if interrupted else "finished", summary={
            "total": len(inputs), "saved_for_review": statuses.count("saved_for_review"),
            "no_recognized_text": statuses.count("no_recognized_text"), "failed": statuses.count("failed"),
            "interrupted": statuses.count("interrupted"), "not_run": len(inputs) - len(rows),
        })
        write_json(output / "report.json", report)
        summary = ["# Thử PP-OCRv6 local", "", "Kết quả chỉ phục vụ kiểm tra, chưa phải dataset đạt yêu cầu.", "",
                   "```json", json.dumps(report["summary"], indent=2), "```", "",
                   "Mỗi ảnh: input, ocr.json, bbox_preview.png và detection_preview.png.",
                   "Nếu có layout: layout/ và regions/ chứa vùng làm việc và JSON gốc từng lượt OCR.",
                   "Số trên preview là chỉ số kết quả OCR, chưa phải thứ tự đọc hoặc mã patch cuối.", ""]
        for row in rows:
            summary.extend([f"- {Path(row['source']).name}: {row['status']}", f"  Kết quả: {row['output_dir']}"])
            if "header_box_count" in row:
                summary.append(f"  Title: {row['header_box_count']}; có chữ: {row['header_recognized_count']}; Notes bỏ qua: {row['notes_region_count']}")
            if row.get("error"):
                summary.append("  Lỗi: " + row["error"].replace("\n", " "))
        with (output / "summary.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(summary) + "\n")
    return report
