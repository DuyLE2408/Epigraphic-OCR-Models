# PaddleOCR V6 + layout trên Colab

Phạm vi: dùng lại pipeline thử V6 đã chạy ở local; không dùng API trường,
segmentation, box repair, alignment hay xuất dataset. Chưa thay pipeline chính.
Một box cho mỗi vùng Title; OCR nguyên dòng tiêu đề, OCR cột trong Content,
bỏ Notes và đưa tọa độ về ảnh gốc. Thứ tự đọc vẫn cần kiểm tra.

## Chạy notebook bằng Run All

Mở https://colab.research.google.com → **Upload** → chọn file
`notebooks/paddle_ocr_v6_colab.ipynb` từ máy bạn. Notebook đã chứa phần kết nối
GitHub; không cần dùng tab GitHub của Colab để tìm/mở repository private.
Mặc định lấy code từ GitHub và dùng Upload để chọn model/ảnh, tải ZIP kết quả về máy.

1. Chọn Runtime → Change runtime type → T4 GPU.
2. Với `CODE_SOURCE=GitHub`, repository **public không cần token**. Notebook thử
   đọc công khai trước; chỉ hỏi xác thực khi không truy cập được repository/nhánh
   hoặc hết hạn mức request chưa đăng nhập. Nếu repo private, thêm Colab Secret
   `GITHUB_TOKEN` có quyền đọc Contents repository này và bật Notebook access,
   hoặc nhập token vào ô ẩn. Không ghi token vào code/URL; token không được lưu trong file.
   Nếu token không truy cập được repo đang cộng tác, chọn `ZIP` và dùng gói code
   mới tạo bằng `scripts/build_colab_paddle_bundle.py` (gói cũ trước notebook thiếu file).
3. Giữ `STORAGE_MODE=Upload` để chạy trước, không cần quyền Google Drive.
   Khi muốn lưu Drive có thể chọn `Drive API`; notebook không gọi `drive.mount` đang lỗi.
4. Chọn `RUN_MODE=One`, `IMAGE_NAME=00099.jpg` để thử một ảnh; `All` chạy input
   trong thư mục input đã chọn, lần lượt từng file. Không quét toàn bộ Drive.
   Nếu chọn ảnh bằng hộp Upload thì One dùng đúng file vừa chọn, kể cả tên khác
   `00099.jpg`; không cần sửa IMAGE_NAME. Tên này chỉ dùng khi tìm ảnh có sẵn.
5. Bấm Runtime → Run all. Lần đầu hoàn tất các ô xác thực/chọn file khi được hỏi.
   Log cài đặt và OCR hiện trực tiếp; người dùng không phải đổi môi trường.

Notebook chỉ lấy danh sách file code cố định qua GitHub Contents API và cố định
toàn bộ file theo cùng một commit SHA của nhánh `Duong`. Không clone corpus hay
pull toàn bộ repository. Code/API trường và thử nghiệm segmentation không được nạp.

Drive mặc định tạo hoặc dùng cấu trúc sau; có thể nhập `DRIVE_ROOT_ID` để dùng một
thư mục dự án có sẵn trong My Drive. Nếu có tên trùng gây mơ hồ thì dừng để chọn lại.

```text
Epigraphic-OCR/
  models/layout/best.pt
  inputs/                 # có thể chia thư mục sách bên trong
  outputs/<run_id>.zip
```

Khi Drive chưa có model/ảnh, notebook dùng file đã upload ở `/content` hoặc mở hộp
chọn file. `SAVE_UPLOADED_ASSETS_TO_DRIVE=True` sao chép model/ảnh đã chọn vào
thư mục dự án, không ghi đè file có sẵn. Model được kiểm tra đúng ba nhãn và chạy
GPU bằng ảnh trắng giả lập trước khi được lưu lên Drive.

Output thử trên máy Colab nằm trong `/content/epigraphic-colab/runs/<session>/results/<run_id>/`.
ZIP chỉ đóng gói kết quả phiên đó và thông tin commit/cấu hình; được lưu vào
Drive `outputs/` hoặc tải về máy ở chế độ Upload. Lỗi một ảnh vẫn có report, các
ảnh khác tiếp tục theo pipeline thử. Nếu lưu Drive lỗi, ZIP còn trên máy Colab;
bật ô tải ZIP dự phòng trước khi kết thúc runtime. Nếu dừng bằng nút Stop/Ctrl+C,
worker được dừng và notebook cố đóng gói phần output đã viết. Không đảm bảo giữ
được file chưa ghi xong khi Colab thu hồi máy đột ngột.

Không cài Paddle vào Python chính của Colab. Notebook giữ phiên bản torch,
torchvision, torchaudio đã có khi cài layout; Paddle dùng `/content/venv-ocr`.
Các thư viện/cache có thể mất khi được cấp máy mới; lần sau notebook tự setup lại.
Mỗi Run All là một lần thử mới, chưa có cơ chế resume của pipeline trường.

## Môi trường đã kiểm tra bởi người dùng

- Colab Python 3.13.16, GPU T4.
- Python chính: PyTorch 2.14.1+cu130; DocLayout-YOLO 0.0.4,
  Ultralytics 8.4.211, dill 0.4.1. Import layout và phép tính PyTorch GPU đã qua.
- `/content/venv-ocr/bin/python`: PaddlePaddle GPU 3.2.0 (cu126),
  PaddleOCR 3.7.0, PaddleX 3.7.2, Pillow 13.0.0, PyMuPDF 1.26.7.
  Import, phép tính Paddle GPU và nạp hai model V6 medium đã qua.
- Chưa xác nhận chạy pipeline trên ảnh thật ở Colab.

Hai môi trường chạy bằng hai tiến trình; không cài Paddle vào Python chính.
NCCL cu12 của Paddle 3.2.0 và cu13 của PyTorch đã gây xung đột khi cài chung.
Không cần người dùng activate/deactivate để chạy pipeline.

## Gói code

Chạy `scripts/build_colab_paddle_bundle.py` bằng Python local để tạo ZIP trong
`data/image_pipeline/colab_bundle/`. Chỉ đóng gói danh sách code cố định cùng
hướng dẫn này và manifest SHA-256; không lấy dữ liệu, token hay model.

Giải nén ZIP vào `/content/paddle-v6/`. Giữ notebook và kết quả ở chế độ riêng tư.
Việc chọn chế độ riêng tư không thay thế quyền đưa tài liệu/model Lab ra ngoài.
Hướng dẫn đồ án giới hạn tài liệu Lab trong phạm vi trường; chưa đưa chúng lên
cloud khi chưa làm rõ quyền sử dụng. Có thể kiểm tra code bằng dữ liệu giả trước.

## Chạy khi đã có model và input được phép sử dụng

Trong notebook, `sys.executable` là Python chính chạy layout. Python OCR nằm ở
`/content/venv-ocr/bin/python`. Không dùng đường dẫn `.venv-layout/Scripts` của Windows.
Ví dụ dưới đây dùng các đường dẫn minh họa, không tự tải hoặc cấp quyền cho dữ liệu:

```python
from pathlib import Path
import subprocess
import sys

command = [
    "/content/venv-ocr/bin/python", "-u", "-B",
    "/content/paddle-v6/run_paddle_ocr_trial.py",
    "--input", "/content/inputs/test.jpg",
    "--device", "gpu:0",
    "--layout-python", sys.executable,
    "--layout-device", "cuda:0",
    "--layout-model", "/content/models/layout/best.pt",
    "--output-root", "/content/paddle-results",
]
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace", bufsize=1)
try:
    for line in process.stdout:
        print(line, end="", flush=True)
    status = process.wait()
except BaseException:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    raise
if status:
    raise RuntimeError(f"Pipeline kết thúc với mã lỗi {status}; xem log/report.")
```

Luồng này tự gọi Python layout, nhận JSON vùng, rồi xử lý bằng Paddle trong
Python OCR. Title và Content đều dùng GPU theo tham số; không tự đổi về CPU.
Ảnh chuẩn bị giữ độ phân giải, detector vẫn có resize nội bộ như local.
Giữ nguyên các ngưỡng/model medium/cách crop, mask và cộng offset của bản local.

Output `/content/paddle-results/<run_id>/` giữ cấu trúc thử local: manifest,
report, summary; mỗi ảnh có input, layout, regions, JSON OCR và hai preview.
Lệnh CLI minh họa trên không tự lưu lên Drive; notebook có bước lưu riêng như đã
mô tả ở đầu tài liệu. Không cập nhật progress của pipeline chính. Mỗi lần
chạy tạo kết quả mới. `/content` và cache model có thể mất khi Colab thu hồi máy;
chủ động lưu kết quả trước khi kết thúc.

Trên Windows, chạy file cũ bằng Run vẫn mặc định CPU và hộp chọn ảnh như trước.
Các tùy chọn mới chỉ đổi thiết bị/thư mục output khi được truyền rõ ràng.
