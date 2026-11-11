"""Kim Han Nom API and resumable local OCR outputs; no local OCR model."""

from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from src.image_batch import _atomic_text, run_all
from src.image_ocr import _write_new_files
from src.image_paths import ocr_folder, prepare_folder, preferred_ocr_path
from src.image_progress import PipelineProgress
from src.page_reference import _file_hash, SUPPORTED_BOOKS, infer_stele_id


BASE_URL = "https://kimhannom.fit.hcmus.edu.vn"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
)
OCR_PARAMETERS = {
    "ocr_id": 1, "lang_type": 1, "reading_direction": 1,
    "epitaph": 1, "tag": None,
}
REQUEST_LIMITS = ((60, 25), (3600, 50), (86400, 200), (604800, 600))


class SchoolAPIError(RuntimeError):
    def __init__(self, message, *, stop_pipeline=False, resume_at=None):
        super().__init__(message)
        self.stop_pipeline = stop_pipeline
        self.resume_at = resume_at


def _json(path):
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SchoolAPIError(f"JSON không hợp lệ: {path.name}")
    return value


def _save_new(path, data):
    content = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_new_files({path: content})


@contextmanager
def _single_run(folder):
    """An OS lock is released even if the process is terminated."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "pipeline.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SchoolAPIError("Đang có một phiên pipeline khác chạy. Hãy đợi phiên đó kết thúc.",
                                 stop_pipeline=True) from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class RequestGate:
    """Count requests across runs; stop at a limit rather than wait for hours."""
    def __init__(self, folder, *, clock=time.time, sleep=time.sleep):
        self.path = folder / "request_history.json"
        self.clock, self.sleep = clock, sleep
        self.calls = 0

    def reserve(self):
        now = self.clock()
        try:
            history = _json(self.path) if self.path.exists() else {"requests": []}
        except (OSError, ValueError, SchoolAPIError):
            raise SchoolAPIError("Không đọc được lịch sử lượt gọi; dừng để kiểm tra hạn mức.",
                                 stop_pipeline=True) from None
        timestamps = history.get("requests")
        cooldown = history.get("cooldown_until", 0)
        if (not isinstance(timestamps, list)
                or any(isinstance(t, bool) or not isinstance(t, (int, float))
                       or not math.isfinite(t) for t in timestamps)
                or isinstance(cooldown, bool) or not isinstance(cooldown, (int, float))
                or not math.isfinite(cooldown)):
            raise SchoolAPIError("Lịch sử lượt gọi bị lỗi; dừng để tránh vượt hạn mức.",
                                 stop_pipeline=True)
        timestamps = sorted(t for t in timestamps if now - t < 604800)
        ready = max(now, cooldown)
        for window, limit in REQUEST_LIMITS:
            recent = [t for t in timestamps if now - t < window]
            if len(recent) >= limit:
                ready = max(ready, recent[-limit] + window + 1)
        if ready > now:
            resume = datetime.fromtimestamp(ready, timezone.utc).isoformat()
            local_time = datetime.fromtimestamp(ready, timezone(timedelta(hours=7))).strftime("%d/%m/%Y %H:%M:%S UTC+7")
            raise SchoolAPIError(
                f"Đã chạm hạn mức API. Đã lưu tiến độ; có thể bấm Run tiếp từ {local_time}.",
                stop_pipeline=True, resume_at=resume)
        if timestamps:
            self.sleep(max(0, timestamps[-1] + 3.1 - now))
        # Reserve before sending: a crash cannot silently erase a spent call.
        timestamps.append(self.clock())
        _atomic_text(self.path, json.dumps({"requests": timestamps, "cooldown_until": cooldown}))
        self.calls += 1

    def cooldown(self, retry_after):
        until = self.clock() + 3600
        if retry_after:
            try:
                seconds = float(retry_after)
                if math.isfinite(seconds):
                    until = max(self.clock() + 1, self.clock() + seconds)
            except ValueError:
                try:
                    until = max(self.clock() + 1, parsedate_to_datetime(retry_after).timestamp())
                except (ValueError, TypeError, OverflowError):
                    pass
        history = _json(self.path)
        history["cooldown_until"] = max(history.get("cooldown_until", 0), until)
        _atomic_text(self.path, json.dumps(history))
        return datetime.fromtimestamp(history["cooldown_until"], timezone.utc).isoformat()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _safe_error(result, token):
    message = result.get("message")
    if isinstance(message, dict):
        message = " — ".join(str(message.get(key) or "") for key in ("title", "text"))
    if not isinstance(message, str):
        return ""
    message = message.replace(token, "[hidden]")
    message = re.sub(r"eyJ[\w-]+\.[\w-]+\.[\w-]+", "[hidden]", message)
    message = re.sub(r"(?i)Bearer\s+\S+", "Bearer [hidden]", message)
    message = re.sub(r"(?i)(token|password|refresh_token)\s*[:=]\s*\S+", r"\1=[hidden]", message)
    message = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email hidden]", message)
    return " ".join(message.split())[:300]


class SchoolClient:
    def __init__(self, token_provider, gate, *, opener=None):
        self.token_provider, self.gate = token_provider, gate
        self.opener = opener or urllib.request.build_opener(_NoRedirect())
        self._token = None

    def _post(self, endpoint, body, content_type):
        if self._token is None:
            token = self.token_provider()
            if not isinstance(token, str) or not token.strip():
                raise SchoolAPIError("Chưa nhập token; tiến độ đã được giữ lại.", stop_pipeline=True)
            token = token.strip()
            if re.search(r"\s|;", token):
                raise SchoolAPIError("Chỉ dán giá trị cookie token, không thêm Bearer hoặc thuộc tính cookie.",
                                     stop_pipeline=True)
            self._token = token
        self.gate.reserve()
        request = urllib.request.Request(
            BASE_URL + endpoint, data=body, method="POST",
            headers={"Authorization": "Bearer " + self._token,
                     "User-Agent": USER_AGENT, "Content-Type": content_type})
        try:
            with self.opener.open(request, timeout=300) as response:
                status = response.status
                content = response.read(20 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status in {401, 403}:
                raise SchoolAPIError(
                    f"HTTP {status}: token hết hạn hoặc tài khoản chưa có quyền. "
                    "Đăng nhập lại, lấy token mới rồi bấm Run tiếp.",
                    stop_pipeline=True) from None
            if status == 429:
                resume = self.gate.cooldown(exc.headers.get("Retry-After"))
                raise SchoolAPIError("HTTP 429: dịch vụ giới hạn lượt gọi. Đã lưu tiến độ.",
                                     stop_pipeline=True, resume_at=resume) from None
            if status == 413:
                raise SchoolAPIError(
                    'HTTP 413: dịch vụ từ chối dung lượng ảnh. Đã giữ ảnh gốc và tiến độ; '
                    'không tự thu nhỏ hoặc gửi lại.', stop_pipeline=True) from None
            detail = ""
            try:
                error = json.loads(exc.read(8192))
                if isinstance(error, dict):
                    detail = _safe_error(error, self._token)
            except (ValueError, OSError):
                pass
            raise SchoolAPIError(f"API trả HTTP {status}. " + detail,
                                 stop_pipeline=300 <= status < 400 or status >= 500) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise SchoolAPIError(
                "Lỗi kết nối hoặc hết thời gian chờ. Chưa xác định máy chủ đã xử lý xong chưa; "
                "đã dừng và giữ tiến độ, không tự gửi lại.", stop_pipeline=True) from None
        if status != 200 or len(content) > 20 * 1024 * 1024:
            raise SchoolAPIError("Phản hồi HTTP hoặc dung lượng JSON không hợp lệ.", stop_pipeline=True)
        try:
            result = json.loads(content)
        except (ValueError, UnicodeError):
            raise SchoolAPIError("API không trả JSON hợp lệ.", stop_pipeline=True) from None
        if (not isinstance(result, dict) or result.get("is_success") is not True
                or result.get("code") != "000000" or not isinstance(result.get("data"), dict)):
            # Do not log server debug payloads: they can contain credentials.
            detail = _safe_error(result, self._token) if isinstance(result, dict) else ""
            raise SchoolAPIError("API báo không thành công trong JSON. " + detail,
                                 stop_pipeline=True)
        if self._token in json.dumps(result, ensure_ascii=False):
            raise SchoolAPIError("Phản hồi chứa thông tin xác thực; không lưu nội dung đó.",
                                 stop_pipeline=True)
        return result

    def upload(self, image):
        boundary = "KimHanNom" + uuid4().hex
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image_file"; '
            f'filename="{image.name}"\r\nContent-Type: image/jpeg\r\n\r\n'
        ).encode("ascii") + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode("ascii")
        result = self._post("/api/web/clc-sinonom/image-upload", body,
                            f"multipart/form-data; boundary={boundary}")
        name = result["data"].get("file_name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise SchoolAPIError("API upload không trả tên ảnh tạm hợp lệ.")
        return name

    def ocr(self, file_name, parameters):
        body = json.dumps({**parameters, "file_name": file_name}).encode("utf-8")
        return self._post("/api/web/clc-sinonom/image-ocr", body, "application/json")


def _parse_boxes(result, image_id):
    data = result.get("data", {})
    entries, texts = data.get("result_bbox"), data.get("result_ocr_text")
    if (result.get("is_success") is not True or result.get("code") != "000000"
            or not isinstance(entries, list) or not isinstance(texts, list)
            or len(entries) != len(texts) or any(not isinstance(t, str) for t in texts)):
        raise SchoolAPIError("Số box/chữ hoặc cấu trúc kết quả OCR không hợp lệ; đã giữ JSON gốc.")
    boxes = []
    for index, (entry, text) in enumerate(zip(entries, texts), 1):
        confidence = None
        polygon = entry
        if isinstance(entry, list) and len(entry) == 2:
            polygon, recognition = entry
            if not isinstance(recognition, list) or len(recognition) != 2 or not isinstance(recognition[0], str):
                raise SchoolAPIError(f"Box {index}: cấu trúc nhận dạng không hợp lệ.")
            # Keep text attached to its polygon, without correcting or sorting it.
            text, confidence = recognition
        if (not isinstance(polygon, list) or len(polygon) != 4
                or any(not isinstance(p, list) or len(p) != 2
                       or any(isinstance(n, bool) or not isinstance(n, (int, float))
                              or not math.isfinite(n) for n in p) for p in polygon)):
            raise SchoolAPIError(f"Box {index}: tọa độ không đủ bốn góc hợp lệ.")
        if confidence is not None and (isinstance(confidence, bool)
                or not isinstance(confidence, (int, float)) or not math.isfinite(confidence)):
            raise SchoolAPIError(f"Box {index}: độ tin cậy không hợp lệ.")
        boxes.append({"image_id": image_id, "patch_id": f"{image_id}.{index:03d}",
                      "bbox": polygon, "sino_nom_ocr": text, "confidence": confidence,
                      "ocr_index": index, "reading_order": None, "patch_path": None,
                      "sino_nom": None})
    return boxes


def _preview(image, boxes):
    with Image.open(image) as source:
        preview = source.convert("RGB")
    draw = ImageDraw.Draw(preview)
    scale = max(1.0, max(preview.size) / 1024)
    font = ImageFont.load_default(size=round(10 * scale))
    offset = round(13 * scale)
    for box in boxes:
        points = [(round(x), round(y)) for x, y in box["bbox"]]
        color = 'orange' if box.get('region') == 'header' else 'lime'
        draw.line(points + [points[0]], fill=color, width=round(2 * scale))
        x = max(0, min(preview.width - round(15 * scale), min(p[0] for p in points)))
        y = max(0, min(preview.height - offset, min(p[1] for p in points) - offset))
        label = str(box["ocr_index"])
        draw.rectangle(draw.textbbox((x, y), label, font=font), fill="black")
        draw.text((x, y), label, fill="yellow", font=font)
    output = BytesIO()
    preview.save(output, format="PNG")
    return output.getvalue()


class SchoolOCR:
    def __init__(self, app_root, client, *, progress=print, parameters=None, pass_name=None):
        self.root, self.client, self.progress = Path(app_root).resolve(), client, progress
        self.parameters = dict(OCR_PARAMETERS if parameters is None else parameters)
        if pass_name not in {None, 'header', 'header/general'}:
            raise ValueError('Unknown OCR pass')
        self.pass_name = pass_name

    def __call__(self, metadata):
        if (metadata.get('processing', {}).get('resolution_policy') != 'native'
                or metadata.get('processing', {}).get('max_dim') is not None
                or metadata.get('resize_scale') != {'x': 1.0, 'y': 1.0}
                or metadata.get('image_size_before_resize') != metadata.get('image_size_after_resize')):
            raise SchoolAPIError('OCR yêu cầu ảnh giữ nguyên độ phân giải; không dùng lại ảnh resize cũ.')
        image = (self.root / metadata["output_jpg"]).resolve()
        image_id = metadata["image_id"]
        folder = ocr_folder(self.root, metadata["book_id"], metadata["stele_id"], image_id)
        if self.pass_name is not None:
            folder = folder / self.pass_name
        images = prepare_folder(self.root, metadata["book_id"], metadata["stele_id"]).resolve()
        if image.parent != images:
            raise SchoolAPIError("Đường dẫn kết quả OCR không thuộc ảnh/sách đang xử lý.")
        digest = _file_hash(image)
        if digest != metadata["output_jpg_sha256"]:
            raise SchoolAPIError("Ảnh chuẩn hóa đã thay đổi; giữ kết quả cũ để kiểm tra.")
        binding = {"image_id": image_id, "image_sha256": digest, "parameters": self.parameters}
        folder.mkdir(parents=True, exist_ok=True)
        manifest = folder / "manifest.json"
        if manifest.exists():
            if _json(manifest) != binding:
                raise SchoolAPIError("Ảnh/cấu hình khác kết quả OCR đã lưu; không ghi đè hoặc tự gọi lại.")
        else:
            allowed = {'header_regions.json'} if self.pass_name is None else set()
            if any(p.name not in allowed for p in folder.iterdir()):
                raise SchoolAPIError("Thư mục OCR có file nhưng thiếu manifest; cần kiểm tra.")
            _save_new(manifest, binding)
        completed = folder / "complete.json"
        files = {"raw": folder / "ocr_raw.json", "boxes": folder / "ocr.json",
                 "preview": folder / "bbox_preview.png"}
        if completed.exists():
            saved = _json(completed)
            if saved.get("binding") != binding or not isinstance(saved.get("sha256"), dict):
                raise SchoolAPIError("Bản ghi hoàn thành OCR không hợp lệ; không tự gửi lại.")
            for key, path in files.items():
                if not path.is_file() or _file_hash(path) != saved["sha256"].get(key):
                    raise SchoolAPIError(f"Kết quả OCR thiếu/hỏng: {path.name}; không tự gọi lại.")
            document = _json(files["boxes"])
            return self._summary(files, len(document["boxes"]), "skipped_existing")
        if not files["raw"].exists():
            upload = folder / "upload.json"
            if upload.exists():
                name = _json(upload).get("file_name")
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                    raise SchoolAPIError("Checkpoint upload không hợp lệ.")
            else:
                self.progress(f"{image_id}: tải ảnh giữ nguyên độ phân giải "
                              f"{metadata['image_size_after_resize']} lên dịch vụ trường.")
                name = self.client.upload(image)
                _save_new(upload, {"file_name": name})
            mode = (' ngang thông thường (epitaph=0) để thử tìm trán bia.'
                    if self.pass_name == 'header/general' else
                    ' ngang để tìm trán bia.' if self.pass_name else ' dọc.')
            self.progress(f"{image_id}: đang OCR" + mode)
            response = self.client.ocr(name, self.parameters)
            _save_new(files["raw"], response)
        result = _json(files["raw"])
        boxes = _parse_boxes(result, image_id)
        document = {"schema_version": "school-ocr-2.0", **binding,
                    "book_id": metadata["book_id"], "image_path": metadata["output_jpg"],
                    "image_size": metadata['image_size_after_resize'],
                    "coordinate_space": "prepared_image_pixels",
                    "reading_order_status": "not_reviewed", "review_required": True,
                    "boxes": boxes}
        if files["boxes"].exists():
            if _json(files["boxes"]) != document:
                raise SchoolAPIError("JSON box hiện có khác JSON gốc; không ghi đè.")
        else:
            _save_new(files["boxes"], document)
        preview = _preview(image, boxes)
        if files["preview"].exists():
            if hashlib.sha256(preview).hexdigest() != _file_hash(files["preview"]):
                raise SchoolAPIError("Ảnh xem box hiện có không khớp JSON; không ghi đè.")
        else:
            _write_new_files({files["preview"]: preview})
        if _file_hash(image) != digest:
            raise SchoolAPIError("Ảnh thay đổi trong khi OCR; chưa ghi nhận hoàn thành.")
        _save_new(completed, {"binding": binding,
                             "sha256": {k: _file_hash(p) for k, p in files.items()}})
        self.progress(f"{image_id}: đã lưu {len(boxes)} box; cần kiểm tra trực quan.")
        return self._summary(files, len(boxes), "success")

    def _summary(self, files, count, status):
        return {"ocr_status": status, "box_count": count, "text_count": count,
                "ocr_json": files["boxes"].relative_to(self.root).as_posix(),
                "ocr_raw_json": files["raw"].relative_to(self.root).as_posix(),
                "bbox_preview": files["preview"].relative_to(self.root).as_posix()}


def run_school_pipeline(app_root, *, token_provider, selected_source=None,
                        book_id=None, stele_id=None, max_dim=None, render_dpi=300,
                        progress=print, recheck_completed=False, include_postprocess=False,
                        include_header=False):
    root = Path(app_root).resolve()
    if max_dim is not None:
        raise ValueError('OCR dùng độ phân giải gốc; không đặt max_dim trước OCR.')
    if selected_source is not None:
        source = Path(selected_source)
        source = (root / source).resolve() if not source.is_absolute() else source.resolve()
        if (book_id not in SUPPORTED_BOOKS or not source.is_file()
                or source.parent != (root / "data/raw" / book_id).resolve()
                or source.suffix.lower() not in {".jpg", ".jpeg", ".pdf"}
                or infer_stele_id(source) != stele_id):
            raise ValueError("Chọn PDF/JPG trực tiếp trong thư mục sách; mã sách/bia phải khớp file nguồn.")
        selected_source = source
    folder = root / "data/image_pipeline/school_api"
    gate = RequestGate(folder)
    client = SchoolClient(token_provider, gate)
    processor = SchoolOCR(root, client, progress=progress)
    if include_header:
        from src.header_ocr import HeaderAwareOCR
        processor = HeaderAwareOCR(root, client, progress=progress)
    box_processor = None
    if include_postprocess:
        from src.box_postprocess import process_ocr_result, record_filter_result

        def box_processor(metadata):
            folder = ocr_folder(root, metadata['book_id'], metadata['stele_id'], metadata['image_id'])
            path = preferred_ocr_path(folder) if include_header else folder / 'ocr.json'
            result = process_ocr_result(root, path)
            record_filter_result(root, result)
            return result

    local_lock = _single_run(root / 'data/image_pipeline/filter_sort') if include_postprocess else nullcontext()
    with _single_run(folder), local_lock:
        state = PipelineProgress(root, OCR_PARAMETERS, max_dim=max_dim, render_dpi=render_dpi,
                                 recheck=recheck_completed,
                                 include_header=include_header,
                                 target_stage='03_filter_sort' if include_postprocess else '02_ocr')
        return run_all(root, max_dim=max_dim, render_dpi=render_dpi,
                       preview=True, progress=progress, ocr_processor=processor, box_processor=box_processor,
                       selected_source=selected_source, progress_store=state)
