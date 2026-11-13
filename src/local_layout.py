"""Run the existing school layout checkpoint in its separate Python environment.

This file is also a standalone worker; it must not import the Paddle pipeline.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from PIL import Image, ImageDraw

LABELS = {"Title", "Content", "Notes"}


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_layout(document, image_size):
    if document.get("image_size") != list(image_size):
        raise ValueError("Layout coordinates do not match the prepared image")
    predictions = document.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Layout predictions must be a list")
    width, height = image_size
    for item in predictions:
        if not isinstance(item, dict) or item.get("label") not in LABELS:
            raise ValueError("Unexpected layout class; expected Title, Content, Notes")
        score, box = item.get("confidence"), item.get("bbox_xyxy")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1
                or not isinstance(box, list) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in box)):
            raise ValueError("Invalid layout score or rectangle")
        if not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height:
            raise ValueError("Layout rectangle is outside the prepared image")
    return document


class LocalLayout:
    def __init__(self, python_executable, model_path, *, confidence=.25, image_size=1024, device="cpu"):
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
            raise ValueError("Layout device must be cpu or cuda:N")
        self.device = device
        self.python = Path(python_executable).resolve()
        self.model = Path(model_path).resolve()
        if not self.python.is_file():
            raise ValueError("Không tìm thấy Python .venv-layout; dùng --no-layout để thử V6 gốc.")
        if not self.model.is_file():
            raise ValueError("Không tìm thấy models/layout/best.pt từ gói trường.")
        self.confidence, self.image_size = confidence, image_size
        self.model_hash = _hash(self.model)

    def describe(self):
        return {"python": str(self.python), "model_path": str(self.model),
                "model_sha256": self.model_hash, "confidence": self.confidence,
                "imgsz": self.image_size, "device": self.device}

    def __call__(self, image, folder):
        output = folder / "layout"
        output.mkdir()
        result_path = output / "layout_regions.json"
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
                   YOLO_AUTOINSTALL="False", HF_HUB_OFFLINE="1")
        command = [str(self.python), "-B", str(Path(__file__).resolve()),
                   "--image", str(image), "--model", str(self.model), "--output", str(result_path),
                   "--confidence", str(self.confidence), "--imgsz", str(self.image_size),
                   "--device", self.device]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                       errors="replace", env=env, timeout=600,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            raise RuntimeError("Layout chạy quá thời gian chờ; đã dừng worker, chưa OCR các vùng.") from None
        with (output / "worker.log").open("x", encoding="utf-8") as stream:
            stream.write(completed.stdout + "\n" + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"Layout worker lỗi; xem {output / 'worker.log'}")
        document = json.loads(result_path.read_text(encoding="utf-8"))
        if document.get("image_sha256") != _hash(image) or document.get("model_sha256") != self.model_hash:
            raise ValueError("Layout image/model checksum mismatch")
        with Image.open(image) as opened:
            validate_layout(document, opened.size)
        return document


def _worker_main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not re.fullmatch(r"cpu|cuda:[0-9]+", args.device):
        raise ValueError("Layout device must be cpu or cuda:N")
    if not args.image.is_file() or not args.model.is_file() or args.output.exists():
        raise ValueError("Missing image/model, or layout output already exists")
    image_hash, model_hash = _hash(args.image), _hash(args.model)
    # The exact school checkpoint was previously trusted/loaded by the user.
    # Scope legacy pickle loading to this isolated process, never the machine.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "False"
    from doclayout_yolo import YOLOv10
    model = YOLOv10(str(args.model))
    if set(model.names.values()) != LABELS:
        raise ValueError("Checkpoint is not the expected Title/Content/Notes layout model")
    result = model.predict(source=str(args.image), imgsz=args.imgsz, conf=args.confidence,
                           device=args.device, verbose=False)[0]
    with Image.open(args.image) as opened:
        preview = opened.convert("RGB")
    predictions = []
    if result.boxes is not None:
        for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(),
                                   result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            predictions.append({"label": model.names[int(cls)], "confidence": float(score),
                                "bbox_xyxy": [float(v) for v in box]})
    document = {"image": args.image.name, "image_size": list(preview.size),
                "image_sha256": image_hash, "model_sha256": model_hash,
                "model": "DocLayout-YOLO (school checkpoint)",
                "parameters": {"imgsz": args.imgsz, "confidence": args.confidence, "device": args.device},
                "predictions": predictions}
    validate_layout(document, preview.size)
    if _hash(args.image) != image_hash or _hash(args.model) != model_hash:
        raise ValueError("Layout input/model changed during inference")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
    draw = ImageDraw.Draw(preview)
    for item in predictions:
        draw.rectangle(item["bbox_xyxy"], outline={"Title": "orange", "Content": "lime", "Notes": "cyan"}[item["label"]], width=5)
        draw.text((item["bbox_xyxy"][0], max(0, item["bbox_xyxy"][1] - 18)), item["label"], fill="orange")
    preview.save(args.output.parent / "layout_preview.png")
    print("LAYOUT_REGIONS:", len(predictions))


if __name__ == "__main__":
    _worker_main()
