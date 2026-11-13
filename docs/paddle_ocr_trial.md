# Thử riêng PaddleOCR V6 trên máy

Colab/GPU: xem [colab_paddle_trial.md](colab_paddle_trial.md). Có thể truyền
`--device gpu:0 --layout-device cuda:0 --layout-python <python-layout>` và
`--output-root <thư-mục-kết-quả>`. Không truyền thì vẫn mặc định CPU như local.

Luồng thử riêng này mặc định dùng **layout trường + PP-OCRv6 local**:
ảnh gốc → Title / Content / Notes → nhận dạng tiêu đề và OCR thân → đưa tọa độ
về ảnh gốc. Chưa ghép vào `run_image_pipeline.py`, chưa sửa box, sắp thứ tự đọc,
crop patches để nộp hay xuất dataset. Không dùng thử nghiệm segmentation.

`--no-layout` giữ chế độ **chỉ PP-OCRv6 trên toàn ảnh** để đối chiếu kết quả gốc.

## 1. Cài môi trường riêng (bạn chạy một lần)

Mở terminal tại thư mục repository `Epigraphic-OCR-Models`.
Nếu `.venv-ocr` đã có như máy hiện tại, dùng ngay môi trường đó. Nếu clone mới,
tạo môi trường bằng Python 3.13 x64 trước:

```powershell
py -3.13 -m venv .venv-ocr
```

Cài bộ phụ thuộc OCR riêng; không cài vào `.venv-image` hoặc `.venv-layout`:

```powershell
.\.venv-ocr\Scripts\python.exe -m pip install -r .\requirements-ocr.txt
```

Kiểm tra engine sau khi cài:

```powershell
.\.venv-ocr\Scripts\python.exe -c "import paddle; print(paddle.__version__); paddle.utils.run_check()"
```

Các phiên bản PaddlePaddle 3.2.0 + paddleocr 3.7.0 đã chạy được trên máy này;
`PP-OCRv6` là tên bản **model**, không phải phiên bản gói Python.
Nếu cài trên máy khác vẫn cần kiểm tra thực tế. `setuptools` được kê khai vì trước
đây môi trường Python 3.13 bị thiếu gói này.

Chế độ layout dùng môi trường `.venv-layout` và `models/layout/best.pt` đã có
trên máy. Chương trình tự gọi worker bằng Python của môi trường này, không yêu
cầu cài PyTorch/DocLayout vào `.venv-ocr`. Nếu clone mới, cần chuẩn bị môi trường
layout và checkpoint trường qua kênh được phép; không lấy model này từ Git.
Worker không cài thêm dependency và chỉ nạp checkpoint trường đã được người dùng
tin cậy. Cách nạp checkpoint cũ qua pickle chỉ đặt trong tiến trình worker.

## 2. Chạy bằng nút Run

1. Trong VS Code chọn **Python: Select Interpreter** → `.venv-ocr/Scripts/python.exe`.
2. Mở `run_paddle_ocr_trial.py` → **Run Python File in Terminal**.
3. Chọn **một ảnh trước**, ví dụ `data/raw/van_khac_han_nom_thang_long_ha_noi/00099.jpg`.
4. Đợi nạp model và OCR. Mở đường dẫn preview/summary được in trong terminal.

Thứ tự chạy mặc định:

1. Layout chạy local trên ảnh đầy đủ: Title / Content / Notes.
2. Với mỗi vùng Title: lấy vùng ảnh trong chương trình, thêm padding nhỏ và dùng
   **TextRecognition** đọc nguyên dòng, giữ **một box theo vùng layout**. Chuỗi
   OCR giữ nguyên; có cờ kiểm tra chiều đọc phải sang trái, không tự đảo ký tự.
3. Với mỗi Content: lấy vùng ảnh và dùng PaddleOCR tìm cột/dòng, nhận dạng chữ.
   Phần Title hoặc Notes chồng lên Content được che trắng trên bản làm việc để
   tránh OCR lặp/đọc ghi chú. Ảnh gốc và bản input đầy đủ không bị che/sửa.
4. Cộng offset crop vào đủ bốn góc box để đưa kết quả về ảnh input đầy đủ.

Không gửi vùng Notes cho OCR. Nếu nhãn Notes bị model dự đoán sai, có thể bỏ mất
chữ thật, nên vẫn phải kiểm tra layout_preview.png. Nếu không tìm thấy Content,
ảnh được báo lỗi, không âm thầm OCR cả ảnh. Nếu không tìm thấy Title, xử lý thân
nhưng báo thiếu trán. Nhiều Title được giữ riêng và báo cần duyệt, không tự gộp.

Hộp chọn cho phép chọn thêm vài ảnh bằng Ctrl khi cần so sánh. Chưa chạy All ba
sách; sau khi đánh giá mẫu mới quyết định tích hợp/batch lớn.

Cách chạy bằng lệnh, cũng tại root:

```powershell
.\.venv-ocr\Scripts\python.exe -B .\run_paddle_ocr_trial.py --input ".\data\raw\van_khac_han_nom_thang_long_ha_noi\00099.jpg"
```

Đối chiếu với V6 gốc bằng cách thêm `--no-layout` vào lệnh trên. Có thể thay
đường dẫn Python/model layout bằng `--layout-python` / `--layout-model`.

Mặc định model medium, chữ Hán phồn thể, CPU 4 luồng. Lần đầu SDK cần Internet
để tải hai model chính thức; ảnh được xử lý trên máy, chương trình không upload
ảnh hoặc gọi API trường. Lần sau dùng model đã lưu trong cache. Có thể chỉ định
hai thư mục model local bằng `--det-model-dir` và `--rec-model-dir`.

Đầu vào: JPG/JPEG, PNG, PDF **một trang**. PDF dùng lại hàm đọc của pipeline ảnh,
ưu tiên trích ảnh nhúng đủ điều kiện, nếu không render 300 DPI. JPG RGB có hướng
đúng được sao chép nguyên byte; trường hợp cần EXIF/RGB lưu bản PNG không mất nét.
Không thu ảnh nguồn xuống 1024. Detector vẫn có bước resize nội bộ của model;
tọa độ kết quả tham chiếu về ảnh đã chuẩn bị. `--det-max-side` cho phép thử thay
đổi giới hạn nội bộ, ví dụ 4096; mỗi cấu hình tạo một kết quả mới để so sánh.

## 3. Output riêng và cách xem

```text
data/image_pipeline/paddle_ocr_trial/<run_id>/
  manifest.json                  # model, phiên bản gói, cấu hình, danh sách input
  report.json                    # tổng kết từng file, lỗi và số box
  summary.md
  001_<tên nguồn>/
    input.jpg hoặc input.png     # bản đưa vào model, giữ kích thước sau EXIF
    input_metadata.json          # nguồn, checksum, kích thước, cách đọc PDF/ảnh
    raw_result.json              # chế độ --no-layout: JSON nguyên bản toàn ảnh
    ocr.json                     # box + chữ + confidence, tiện tích hợp sau
    bbox_preview.png             # các box có kết quả recognition
    detection_preview.png        # mọi box detection, kể cả box không có chữ
    error.json                   # chỉ có nếu file lỗi
    layout/                      # chỉ có trong chế độ layout
      layout_regions.json
      layout_preview.png         # Title cam / Content xanh lá / Notes xanh lam
      worker.log
    regions/
      title_001/
        input.png                # crop làm việc, chưa phải patch để nộp
        crop_metadata.json       # tọa độ nguồn, padding, phần che nếu chồng Notes
        raw_result.json          # kết quả recognition nguyên tiêu đề
      content_001/
        input.png                # crop làm việc, che phần chồng Title/Notes
        crop_metadata.json
        raw_result.json          # kết quả detection + recognition của thân
        ocr_in_crop.json          # tọa độ trước khi cộng offset
    excluded_boxes.json          # box dư rơi vào Title/Notes sau OCR thân, nếu có
```

Mỗi lần chạy tạo thư mục mới, không ghi đè kết quả cũ. Đây là chạy thử lại có chủ
đích, chưa dùng progress.json của pipeline chính; cũng không cập nhật lịch sử API.

`ocr.json` giữ `rec_polys` đi cùng `rec_texts`, `rec_scores` đối với thân/toàn ảnh.
Box tiêu đề có nguồn **layout**, chữ có nguồn **recognition**; không coi box
layout là box ký tự do OCR tự phát hiện. Detection chưa lọc
nằm riêng, tránh ghép nhầm chữ với `dt_polys` khi số lượng khác nhau.
Ngưỡng recognition mặc định 0.0 để không ẩn mất box có chữ dự đoán yếu trong
thử nghiệm. Kết quả yếu vẫn cần kiểm tra, không tự nâng thành nhãn chuẩn.

Trên bbox_preview.png: tiêu đề cam, nhãn **Title**; thân xanh lá, có chỉ số để
đối chiếu JSON. Các số **chưa phải thứ tự đọc**. Chế độ layout gán `header`/`body`;
chế độ V6 gốc để vùng `unclassified`. `reading_order`, `patch_id`, `patch_path`,
`sino_nom` vẫn để null.
ID thử dùng tên nguồn, chưa phải mã dataset để nộp. OCR có thể đọc cả ghi chú,
thiếu trán, tách cột hoặc sai chữ; `saved_for_review` chỉ nghĩa lưu kết quả thành
công. `no_recognized_text` được báo riêng, không coi là OCR đạt.

Kiểm tra lần đầu: trán có box/chữ không; thân có thiếu/chia cột không; box có ôm
đủ nét; đối chiếu hai preview để biết detection thiếu hay recognition không trả
chữ. Không in nội dung chữ Hán trong báo cáo terminal; xem JSON local nếu cần.

V6 gốc đã được người dùng chạy thật trên bia 99: 21 box, chỉ phát hiện 2/4 chữ
trán, có ghi chú/vùng thừa. Chế độ layout được bổ sung sau lần thử đó; kiểm thử
luồng bằng dữ liệu giả không chứng minh chất lượng nhận dạng trán/thân ảnh thật.
Chưa đánh giá accuracy của chữ OCR hoặc bảo đảm layout đúng trên mọi bia.
