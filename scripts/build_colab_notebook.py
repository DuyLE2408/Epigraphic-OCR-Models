"""Generate the source-only Colab notebook; never embed local outputs or secrets."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def notebook():
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "id": f"cell-{len(cells):02d}",
                      "metadata": {}, "source": dedent(text).strip().splitlines(keepends=True)})

    def code(text):
        cells.append({"cell_type": "code", "id": f"cell-{len(cells):02d}", "metadata": {},
                      "execution_count": None, "outputs": [],
                      "source": dedent(text).strip().splitlines(keepends=True)})

    md('''
    # Epigraphic OCR — PaddleOCR V6 + nhận diện vùng

    Luồng: **ảnh gốc → Title / Content / Notes → OCR nguyên tiêu đề + OCR thân → JSON và preview**.
    Giữ một box cho mỗi vùng Title, bỏ Notes, đưa tọa độ về ảnh gốc. Đây là inference
    bằng model có sẵn, chưa phải train. Chưa sửa box, chốt thứ tự đọc, tạo patches hay dataset.

    **Mở notebook:** vào https://colab.research.google.com → **Upload** → chọn file
    `.ipynb` này từ máy bạn. Không cần mở notebook qua tab GitHub của Colab.

    **Cách chạy:** Runtime → Change runtime type → **T4 GPU**, giữ cấu hình mặc định,
    rồi **Runtime → Run all**. Notebook tự gọi hai môi trường; không cần activate thủ công.
    Chọn model/ảnh khi được hỏi rồi chờ chạy tiếp. Repo public không cần token GitHub.

    Notebook thử đọc code public trước. Chỉ khi không truy cập được repository/nhánh
    mới thử xác thực. Với repository private: thêm **Colab Secrets** tên `GITHUB_TOKEN`, bật Notebook access,
    dùng token được phép **đọc Contents của repository này**. Có thể nhập bằng ô ẩn nếu
    chưa có Secret. Không viết token vào code hoặc gửi vào chat. Nếu không dùng token,
    chọn `CODE_SOURCE = "ZIP"` và upload gói code tạo bởi `scripts/build_colab_paddle_bundle.py`.

    Mặc định `GitHub` lấy code nhánh `Duong`, `Upload` chọn model/ảnh từ máy bạn và
    tải ZIP kết quả về máy. Không cần kết nối Drive để chạy luồng mặc định.
    `Drive API` là tùy chọn lưu trữ bổ sung, không dùng `drive.mount` đã bị lỗi.
    Chỉ chọn ảnh/model được phép đưa lên cloud; chế độ riêng tư không thay thế quy định Lab.
    Notebook không chứa hoặc tự tải tài liệu, ảnh bia, model layout hay token từ GitHub.
    ''')
    code('''
    # @title 1. Cấu hình
    from pathlib import Path

    CODE_SOURCE = "GitHub" # @param ["GitHub", "ZIP"]
    STORAGE_MODE = "Upload" # @param ["Upload", "Drive API"]
    RUN_MODE = "One" # @param ["One", "All"]
    IMAGE_NAME = "00099.jpg" # @param {type:"string"}
    DRIVE_ROOT_ID = "" # @param {type:"string"}
    LOCAL_INPUT_DIR = "" # @param {type:"string"}
    SAVE_UPLOADED_ASSETS_TO_DRIVE = True # @param {type:"boolean"}
    SHOW_PREVIEW = True # @param {type:"boolean"}

    REPOSITORY = "DuyLE2408/Epigraphic-OCR-Models"
    BRANCH = "Duong"
    WORK_ROOT = Path("/content/epigraphic-colab")
    OCR_PYTHON = Path("/content/venv-ocr/bin/python")
    REQUIRED_FILES = (
        "run_paddle_ocr_trial.py", "src/paddle_ocr_trial.py",
        "src/paddle_layout_ocr.py", "src/local_layout.py",
        "src/image_ocr.py", "src/image_paths.py", "src/colab_runtime.py",
        "requirements-colab-layout.txt", "requirements-colab-ocr.txt",
    )
    ''')
    code('''
    # @title 2. Lấy riêng code từ nhánh Duong (không clone dữ liệu)
    import base64
    import getpass
    import hashlib
    import json
    from uuid import uuid4
    import zipfile

    CODE_ROOT = WORK_ROOT / "code" / uuid4().hex[:12]
    CODE_ROOT.mkdir(parents=True, exist_ok=False)

    def load_github_code():
        import requests
        session = requests.Session()
        session.headers.update({"Accept": "application/vnd.github+json"})
        def request(path, *, authenticate_if_needed=False, **kwargs):
            response = session.get("https://api.github.com/repos/" + REPOSITORY + path,
                                   timeout=(15, 90), allow_redirects=False, **kwargs)
            if authenticate_if_needed and response.status_code in (401, 403, 404):
                from google.colab import userdata
                try:
                    token = userdata.get("GITHUB_TOKEN")
                except Exception:
                    token = ""
                if not token:
                    print("Chưa đọc được repo/nhánh public (hoặc đã chạm hạn mức GitHub).")
                    token = getpass.getpass("Token GitHub nếu repo private; bỏ trống để dừng: ")
                if not token.strip():
                    raise RuntimeError("Kiểm tra repo/nhánh hoặc chọn CODE_SOURCE=ZIP; chưa có code để chạy.")
                session.headers.update({"Authorization": "Bearer " + token.strip()})
                response = session.get("https://api.github.com/repos/" + REPOSITORY + path,
                                       timeout=(15, 90), allow_redirects=False, **kwargs)
            if response.status_code != 200:
                raise RuntimeError(f"GitHub HTTP {response.status_code}; kiểm tra quyền đọc repo/nhánh. "
                                   "Có thể chọn CODE_SOURCE=ZIP.")
            return response.json()
        try:
            sha = request("/git/ref/heads/" + BRANCH, authenticate_if_needed=True)["object"]["sha"]
            for name in REQUIRED_FILES:
                value = request("/contents/" + name, params={"ref": sha})
                if value.get("type") != "file" or value.get("encoding") != "base64":
                    raise RuntimeError("GitHub không trả về file code đúng định dạng: " + name)
                data = base64.b64decode(value["content"].replace("\\n", ""), validate=True)
                digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\\0" + data).hexdigest()
                if digest != value["sha"]:
                    raise RuntimeError("Checksum code không khớp: " + name)
                destination = CODE_ROOT / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            return sha
        finally:
            session.headers.pop("Authorization", None)
            session.close()

    if CODE_SOURCE == "GitHub":
        SOURCE_VERSION = load_github_code()
    elif CODE_SOURCE == "ZIP":
        from google.colab import files
        upload_folder = CODE_ROOT / "upload"
        upload_folder.mkdir()
        uploaded = files.upload(target_dir=str(upload_folder))
        archives = [Path(p) for p in uploaded if Path(p).suffix.lower() == ".zip"]
        del uploaded
        if len(archives) != 1:
            raise ValueError("Chọn đúng một gói ZIP code mới của pipeline V6.")
        with zipfile.ZipFile(archives[0]) as archive:
            manifest = json.loads(archive.read("bundle_manifest.json"))
            for name in REQUIRED_FILES:
                data = archive.read(name)
                if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
                    raise RuntimeError("Checksum ZIP không khớp: " + name)
                destination = CODE_ROOT / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
        SOURCE_VERSION = "code-zip"
    else:
        raise ValueError("CODE_SOURCE phải là GitHub hoặc ZIP")
    print("CODE_READY:", SOURCE_VERSION)
    ''')
    code('''
    # @title 3. Chọn nơi lưu (xác thực Drive trước khi cài model)
    import importlib.util
    spec = importlib.util.spec_from_file_location("epigraphic_colab_runtime", CODE_ROOT / "src/colab_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    workspace = runtime.new_workspace(WORK_ROOT / "runs")
    storage = None

    if STORAGE_MODE == "Drive API":
        from google.colab import auth
        import google.auth
        from googleapiclient.discovery import build
        auth.authenticate_user()
        credentials, _ = google.auth.default()
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        _ = service.about().get(fields="storageQuota").execute()
        storage = runtime.DriveStorage(service)
        if DRIVE_ROOT_ID.strip():
            drive_root = DRIVE_ROOT_ID.strip()
            info = service.files().get(fileId=drive_root, fields="mimeType").execute()
            if info["mimeType"] != "application/vnd.google-apps.folder":
                raise ValueError("DRIVE_ROOT_ID phải là ID thư mục dự án trên My Drive")
        else:
            drive_root = storage.folder("Epigraphic-OCR")
        drive_models = storage.folder("layout", storage.folder("models", drive_root))
        drive_inputs = storage.folder("inputs", drive_root)
        drive_outputs = storage.folder("outputs", drive_root)
        print("DRIVE_API_OK. Thư mục dự án:", "https://drive.google.com/drive/folders/" + drive_root)
    elif STORAGE_MODE != "Upload":
        raise ValueError("Chọn Drive API hoặc Upload")
    print("WORKSPACE:", workspace)
    ''')
    code('''
    # @title 4. Chuẩn bị hai môi trường và kiểm tra GPU tự động
    import sys
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Chưa có GPU. Runtime → Change runtime type → T4 GPU.")
    print("GPU:", torch.cuda.get_device_name(0))
    runtime.setup_environments(CODE_ROOT, OCR_PYTHON)
    print("ENVIRONMENTS_READY")
    ''')
    md('''
    ## Model layout và ảnh đầu vào

    Drive có cấu trúc `Epigraphic-OCR/models/layout/best.pt`, `inputs/`, `outputs/`.
    Có thể đặt các thư mục sách bên trong `inputs/`; One tìm đúng một file theo tên
    hoặc đường dẫn tương đối, ví dụ `ten_sach/00099.jpg` để phân biệt tên trùng.
    All chỉ xử lý ảnh JPG/PNG hoặc PDF một trang trong thư mục input này.
    Khi chọn file bằng hộp Upload, One dùng đúng ảnh vừa chọn; không cần sửa IMAGE_NAME.
    IMAGE_NAME chỉ dùng để tìm ảnh có sẵn trong Drive hoặc LOCAL_INPUT_DIR.

    Nếu Drive chưa có model/ảnh, notebook dùng file đã upload ở `/content` hoặc mở
    hộp chọn file. Với model, chọn **models/layout/best.pt** đang dùng ở local.
    Khi bật `SAVE_UPLOADED_ASSETS_TO_DRIVE`, model và ảnh đã chọn được sao chép lên
    thư mục dự án sau khi model qua kiểm tra. Notebook không ghi đè file có sẵn.
    ''')
    code('''
    # @title 5. Lấy và kiểm tra checkpoint Title / Content / Notes
    import shutil
    from google.colab import files
    model_path = workspace / "model/best.pt"
    model_on_drive = [] if storage is None else [r for r in storage.children(drive_models) if r["name"] == "best.pt"]
    if len(model_on_drive) > 1:
        raise ValueError("Drive có nhiều best.pt trùng tên; giữ đúng một model layout.")
    if model_on_drive:
        storage.download(model_on_drive[0]["id"], model_path)
    elif Path("/content/best.pt").is_file():
        shutil.copyfile("/content/best.pt", model_path)
    else:
        print("Chọn models/layout/best.pt từ máy bạn:")
        uploaded = files.upload(target_dir=str(workspace / "model/upload"))
        selected = [Path(p) for p in uploaded if Path(p).suffix.lower() == ".pt"]
        del uploaded
        if len(selected) != 1:
            raise ValueError("Cần đúng một checkpoint layout .pt")
        shutil.copyfile(selected[0], model_path)

    model_check = """
    import os, sys
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    os.environ['YOLO_AUTOINSTALL']='False'
    import numpy as np
    import torch
    from doclayout_yolo import YOLOv10
    model=YOLOv10(sys.argv[1])
    assert set(model.names.values())=={'Title','Content','Notes'}, 'Sai model layout'
    with torch.inference_mode():
        model.predict(np.full((1024,1024,3),255,dtype=np.uint8),device='cuda:0',imgsz=1024,verbose=False,save=False)
    print('LAYOUT_GPU_READY:', next(model.model.parameters()).device)
    """
    from textwrap import dedent
    runtime.stream_command([sys.executable, "-u", "-c", dedent(model_check), model_path])
    if storage is not None and SAVE_UPLOADED_ASSETS_TO_DRIVE and not model_on_drive:
        storage.upload(model_path, drive_models)
        print("Đã lưu model layout vào Drive.")
    ''')
    code('''
    # @title 6. Chọn một ảnh hoặc All
    sources = []
    from_drive = False
    selected_by_upload = False
    if storage is not None:
        entries = storage.input_files(drive_inputs)
        if RUN_MODE == "One":
            entries = [r for r in entries if runtime.matches_input_name(r["relative_path"], IMAGE_NAME)]
            if len(entries) > 1:
                raise ValueError("Nhiều ảnh trùng tên; nhập IMAGE_NAME dạng ten_sach/00099.jpg.")
        if entries:
            for row in entries:
                sources.append(storage.download(row["id"], workspace / "inputs" / row["relative_path"]))
            from_drive = True
    if not sources:
        if LOCAL_INPUT_DIR.strip():
            local_dir = Path(LOCAL_INPUT_DIR)
            if not local_dir.is_dir():
                raise ValueError("LOCAL_INPUT_DIR không tồn tại")
            sources = [p for p in local_dir.rglob("*") if p.is_file() and not p.is_symlink()]
        elif RUN_MODE == "One" and Path(IMAGE_NAME).name == IMAGE_NAME and (Path("/content") / IMAGE_NAME).is_file():
            sources = [Path("/content") / IMAGE_NAME]
        else:
            print("Chọn ảnh/PDF một trang cần xử lý:")
            uploaded = files.upload(target_dir=str(workspace / "inputs"))
            sources = [Path(p) for p in uploaded]
            del uploaded
            selected_by_upload = True
    sources = runtime.select_inputs(sources, RUN_MODE, IMAGE_NAME, selected_by_upload=selected_by_upload)
    if storage is not None and SAVE_UPLOADED_ASSETS_TO_DRIVE and not from_drive:
        # Preflight all names before uploading, so ambiguous names do not cause partial copies.
        existing_names = {r['name'] for r in storage.children(drive_inputs)}
        names = [p.name for p in sources]
        if len(names) != len(set(names)) or existing_names.intersection(names):
            raise ValueError("Input Drive có tên trùng; tổ chức lại thư mục sách trước khi lưu.")
        for path in sources:
            storage.upload(path, drive_inputs)
    print("Số input được chọn:", len(sources))
    print("Chế độ:", RUN_MODE)
    print("Ảnh đầu tiên:", sources[0].name)
    ''')
    code('''
    # @title 7. Chạy pipeline — tự gọi đúng môi trường, hiện log trực tiếp
    command = runtime.pipeline_command(CODE_ROOT, OCR_PYTHON, sys.executable,
                                       model_path, sources, workspace / "results")
    pipeline_exit = None
    interrupted = False
    try:
        pipeline_exit = runtime.stream_command(command, check=False)
    except KeyboardInterrupt:
        interrupted = True
        print("Đã dừng pipeline. Đang giữ và đóng gói kết quả đã viết.")
    finally:
        provenance = {"repository": REPOSITORY, "branch": BRANCH, "source_version": SOURCE_VERSION,
                      "mode": RUN_MODE, "input_count": len(sources), "exit_code": pipeline_exit,
                      "interrupted": interrupted, "review_required": True}
        (workspace / "results/colab_run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        result_zip = runtime.archive_results(workspace / "results", workspace / f"{workspace.name}.zip")
        print("ZIP kết quả local:", result_zip)
        try:
            if storage is not None:
                result_id = storage.upload(result_zip, drive_outputs)
                print("Đã lưu kết quả Drive:", "https://drive.google.com/file/d/" + result_id + "/view")
            else:
                files.download(str(result_zip))
        except Exception as save_error:
            print("Lưu Drive/tải xuống chưa thành công:", type(save_error).__name__)
            print("ZIP vẫn còn tại:", result_zip)
            print("Chạy ô tải lại kết quả phía dưới trước khi kết thúc runtime.")
    print("PIPELINE_EXIT:", pipeline_exit)
    if pipeline_exit not in (0, None):
        print("Có input lỗi hoặc chưa có chữ nhận dạng. Xem report.json / summary.md trong ZIP.")
    ''')
    code('''
    # @title 8. Tổng kết và xem tối đa hai ảnh preview
    reports = sorted((workspace / "results").glob("*/report.json"))
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        for row in report["files"]:
            if row["status"] == "failed":
                print("Lỗi:", Path(row["source"]).name, row.get("error_type"), row.get("error"))
    if SHOW_PREVIEW:
        from PIL import Image
        from IPython.display import display
        for preview in sorted((workspace / "results").glob("*/*/bbox_preview.png"))[:2]:
            with Image.open(preview) as image:
                view = image.copy()
            view.thumbnail((1100, 1400))  # Chỉ thu nhỏ phần hiển thị, không sửa ảnh kết quả.
            display(view)
    print("Kết quả cần kiểm tra trực quan; số box chưa phải thứ tự đọc/patch cuối.")
    ''')
    md('''
    ## Tải lại kết quả nếu lần lưu cuối gặp lỗi

    Ô dưới không chạy OCR lại. Chỉ chạy khi cần tải ZIP đã tạo về máy.
    Với `Drive API`, ZIP bình thường đã nằm trong `Epigraphic-OCR/outputs/`.

    Mỗi lần Run All tạo một thư mục thử mới; chưa nối progress.json của pipeline chính.
    File ở `/content` và các thư viện có thể mất khi Colab cấp máy mới. Drive giữ model,
    input đã lưu và output; lần sau notebook tự cài lại môi trường khi cần.
    ''')
    code('''
    # @title Tải ZIP thủ công (bật khi cần)
    DOWNLOAD_AGAIN = False # @param {type:"boolean"}
    if DOWNLOAD_AGAIN:
        from google.colab import files
        files.download(str(result_zip))
    ''')
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "gpuClass": "standard",
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"},
                         "colab": {"name": "paddle_ocr_v6_colab.ipynb", "provenance": []}},
            "cells": cells}


if __name__ == "__main__":
    destination = ROOT / "notebooks/paddle_ocr_v6_colab.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(destination)
