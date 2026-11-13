"""Small Colab helpers; no model imports, credentials on disk or school OCR API."""

from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
from uuid import uuid4
import zipfile


INPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf"}


def new_workspace(parent):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(parent) / f"{stamp}_{uuid4().hex[:8]}"
    for folder in ("inputs", "model", "results"):
        (root / folder).mkdir(parents=True, exist_ok=False)
    return root


def stream_command(command, *, cwd=None, check=True):
    """Stream output and stop the full POSIX worker group on notebook interrupt."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               PYTHONDONTWRITEBYTECODE="1")
    # A notebook token stays in the parent session, never in a model subprocess.
    for name in tuple(env):
        if name in {"GITHUB_TOKEN", "GH_TOKEN", "KAGGLE_API_TOKEN"}:
            env.pop(name)
    process = subprocess.Popen(
        [str(p) for p in command], cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        start_new_session=(os.name == "posix"),
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    except BaseException:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()
    if check and code:
        raise RuntimeError(f"Lệnh kết thúc với mã {code}; xem log phía trên.")
    return code


def setup_environments(code_root, ocr_python="/content/venv-ocr/bin/python"):
    """Reuse Colab torch; isolate Paddle's cu12/NCCL packages from base Python."""
    code_root, ocr_python = Path(code_root), Path(ocr_python)
    # Never let layout dependency resolution replace the preinstalled torch stack.
    pins = []
    for package in ("torch", "torchvision", "torchaudio"):
        try:
            pins.append(f"{package}=={version(package)}")
        except Exception:
            if package == "torch":
                raise RuntimeError("Chọn Colab GPU runtime có sẵn PyTorch trước.") from None
    constraints = code_root / "colab_torch_constraints.txt"
    constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
    stream_command([sys.executable, "-m", "pip", "install", "-c", constraints,
                    "-r", code_root / "requirements-colab-layout.txt"])
    if not ocr_python.is_file():
        stream_command([sys.executable, "-m", "pip", "install", "virtualenv==21.15.6"])
        stream_command([sys.executable, "-m", "virtualenv", ocr_python.parent.parent])
    stream_command([ocr_python, "-m", "pip", "install", "setuptools>=75", "wheel"])
    stream_command([ocr_python, "-m", "pip", "install", "paddlepaddle-gpu==3.2.0",
                    "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/"])
    stream_command([ocr_python, "-m", "pip", "install", "-r", code_root / "requirements-colab-ocr.txt"])
    stream_command([sys.executable, "-u", "-c", "import torch; from doclayout_yolo import YOLOv10; "
                    "assert torch.cuda.is_available(), 'GPU unavailable'; "
                    "x=torch.ones(1,device='cuda'); print('LAYOUT_ENV_GPU_OK:',x.item())"])
    stream_command([ocr_python, "-u", "-c", "import paddle; import paddleocr; "
                    "assert paddle.is_compiled_with_cuda(); paddle.set_device('gpu:0'); "
                    "x=paddle.ones([1]); print('OCR_ENV_GPU_OK:',x.place,x.numpy().tolist())"])
    return ocr_python


def matches_input_name(path, image_name):
    if not isinstance(image_name, str) or not image_name:
        raise ValueError("Nhập tên ảnh hoặc đường dẫn tương đối trong inputs")
    requested = PurePosixPath(image_name.replace("\\", "/"))
    if requested.is_absolute() or ".." in requested.parts or not requested.name:
        raise ValueError("Tên ảnh phải là đường dẫn tương đối bên trong inputs")
    return Path(path).parts[-len(requested.parts):] == requested.parts


def select_inputs(paths, mode, image_name="00099.jpg", *, selected_by_upload=False):
    candidates = sorted(Path(p) for p in paths if Path(p).suffix.lower() in INPUT_SUFFIXES)
    if mode == "One":
        if not selected_by_upload:
            candidates = [p for p in candidates if matches_input_name(p, image_name)]
        if len(candidates) != 1:
            if selected_by_upload:
                raise ValueError(f"Chế độ One cần chọn đúng một ảnh/PDF; đã chọn {len(candidates)} file hợp lệ.")
            raise ValueError(f"Cần đúng một ảnh tên {image_name}; tìm được {len(candidates)}. Kiểm tra input.")
    elif mode != "All":
        raise ValueError("Chế độ phải là One hoặc All")
    if not candidates:
        raise ValueError("Chưa có ảnh/PDF đầu vào để chạy")
    return candidates


def pipeline_command(code_root, ocr_python, layout_python, model, sources, output_root):
    return [str(ocr_python), "-u", "-B", str(Path(code_root) / "run_paddle_ocr_trial.py"),
            "--input", *(str(p) for p in sources), "--device", "gpu:0",
            "--layout-python", str(layout_python), "--layout-device", "cuda:0",
            "--layout-model", str(model), "--output-root", str(output_root)]


def archive_results(results, destination):
    """Archive only this run's result directory; never collect input collections."""
    results, destination = Path(results).resolve(), Path(destination).resolve()
    if destination.is_relative_to(results):
        raise ValueError("ZIP phải nằm ngoài thư mục kết quả được nén")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(results.rglob("*")):
            if path.is_symlink():
                raise ValueError("Không đóng gói symlink trong output")
            if path.is_file():
                archive.write(path, path.relative_to(results).as_posix())
    return destination


def _drive_name(name):
    if not isinstance(name, str) or not name or name in {".", ".."} or any(c in name for c in "/\\\x00"):
        raise ValueError("Tên file/thư mục Drive không hợp lệ")
    return name


class DriveStorage:
    """Explicit folder operations via Drive API. No mount, sharing or deletion."""
    def __init__(self, service):
        self.service = service

    def children(self, parent):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", parent):
            raise ValueError("Drive folder ID không hợp lệ")
        token = None
        rows = []
        while True:
            page = self.service.files().list(
                q=f"'{parent}' in parents and trashed = false", pageSize=1000,
                fields="nextPageToken,files(id,name,mimeType,size)", pageToken=token,
            ).execute(num_retries=3)
            rows.extend(page.get("files", []))
            token = page.get("nextPageToken")
            if not token:
                return rows

    def folder(self, name, parent="root"):
        name = _drive_name(name)
        matches = [r for r in self.children(parent) if r["name"] == name]
        if len(matches) > 1:
            raise ValueError(f"Có nhiều mục Drive trùng tên {name}; chọn một folder ID rõ ràng.")
        if matches:
            if matches[0]["mimeType"] != "application/vnd.google-apps.folder":
                raise ValueError(f"Mục {name} trên Drive không phải thư mục")
            return matches[0]["id"]
        return self.service.files().create(body={"name": name, "parents": [parent],
                    "mimeType": "application/vnd.google-apps.folder"}, fields="id").execute()["id"]

    def download(self, file_id, destination):
        from googleapiclient.http import MediaIoBaseDownload
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with destination.open("xb") as stream:
                created = True
                downloader = MediaIoBaseDownload(stream, self.service.files().get_media(fileId=file_id),
                                                chunksize=8 * 1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk(num_retries=3)
        except FileExistsError:
            raise
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise
        return destination

    def upload(self, source, parent, *, name=None):
        from googleapiclient.http import MediaFileUpload
        source = Path(source)
        name = _drive_name(name or source.name)
        if any(r["name"] == name for r in self.children(parent)):
            raise FileExistsError(f"Drive đã có {name}; không ghi đè")
        request = self.service.files().create(
            body={"name": name, "parents": [parent]},
            media_body=MediaFileUpload(str(source), resumable=True, chunksize=8 * 1024 * 1024),
            fields="id",
        )
        result = None
        while result is None:
            _, result = request.next_chunk(num_retries=3)
        return result["id"]

    def input_files(self, parent, relative=Path(), seen=None):
        seen = set() if seen is None else seen
        if parent in seen:
            raise ValueError("Folder Drive bị lặp")
        seen.add(parent)
        rows = []
        for row in self.children(parent):
            name = _drive_name(row["name"])
            path = relative / name
            if row["mimeType"] == "application/vnd.google-apps.folder":
                rows.extend(self.input_files(row["id"], path, seen))
            elif Path(name).suffix.lower() in INPUT_SUFFIXES and not row["mimeType"].startswith("application/vnd.google-apps."):
                rows.append({**row, "relative_path": path.as_posix()})
        return rows
