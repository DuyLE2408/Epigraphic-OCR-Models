# Bàn giao Thành viên 2 — 11/11/2026

Đọc file này cùng `docs/image_pipeline.md` trước khi làm tiếp. Mục tiêu hiện tại
là giải quyết trán bia; chưa chuyển sang viết bước crop hoặc xuất dataset.

## Cách chạy và phạm vi

- Mở repository root trong VS Code, chọn `.venv-image/Scripts/python.exe`,
  mở `run_image_pipeline.py` rồi Run. Có chế độ Chọn 1 ảnh, All, và xử lý box cục bộ.
- Input PDF một trang/JPG nằm trực tiếp trong `data/raw/<book_id>/` của ba sách.
- `requirements-image.txt` chỉ cần Pillow và PyMuPDF; API dùng thư viện chuẩn Python.
- Clone mới cần tự tạo môi trường và nhận dữ liệu qua kênh Lab cho phép.
- Không sửa code Thành viên 1/3, `main.py`, dữ liệu gốc hoặc `processed_text`.
- Không gửi tài liệu/ảnh Lab sang dịch vụ ngoài. School API đã được người dùng chọn;
  không tự gọi API ảnh thật, cài dependency hay chạy model trong khi chỉ phân tích.
- Người dùng muốn tự chạy ảnh thật và theo dõi từng bước.

## Pipeline đang có

`raw -> 01_prepare -> 02_ocr -> 03_filter_sort`.

- 01: EXIF/RGB/JPG, giữ độ phân giải. JPG RGB có hướng đúng được sao chép nguyên byte.
  PDF được trích ảnh nhúng khi đủ điều kiện, nếu không render 300 DPI.
  Không thu nhỏ xuống 1024 px trước OCR.
- 02: API Kim Hán Nôm, OCR dọc toàn ảnh và OCR ngang toàn ảnh; không crop trước upload.
  Token chỉ lấy từ biến môi trường `KIMHANNOM_TOKEN` hoặc hộp thoại nhập ẩn.
- 03: lọc box, sửa hình học thận trọng, sắp trán phải sang trái; thân trái sang phải,
  trên xuống dưới theo hướng dẫn. Chữ OCR không phải ground truth.
- Chưa có patches, alignment hoặc xuất dataset mới. Ảnh trong `dataset/images`
  phải là ảnh sạch; bbox nằm trong nhãn riêng. Khi xuất ảnh nhỏ phải đổi tọa độ tương ứng.
- Patches sau này crop từ ảnh đầy đủ độ phân giải; cột đứng xoay 90 độ ngược chiều
  kim đồng hồ, chiều cao cuối tối đa 96 px. Số trang chỉ giữ giá trị được cung cấp.

Output theo `data/image_pipeline/<book_id>/<stele_id>/`:

```text
01_prepare/<image_id>.jpg, .json, .preview.html
02_ocr/<image_id>/
  ocr_raw.json, ocr.json, bbox_preview.png, manifest/upload/complete.json
  header/general/                  # lượt ngang hiện tại
  ocr_combined.json, bbox_combined_preview.png
  header_selection.json, combined_sources.json, complete_combined.json
03_filter_sort/<image_id>/
  rules.json, boxes.json, bbox_preview.png, proposals_preview.png
  report.json, complete.json, _history/
```

`school_api/progress.json` lưu hàng đợi; `request_history.json` lưu hạn mức.
Hết hạn mức phải lưu và dừng, lần Run sau tiếp tục; không tự chờ nhiều giờ.
Không xóa cache/progress để thử lại một thay đổi bộ lọc: code dùng lại OCR đã lưu.
All vẫn đọc tên file để phát hiện input mới, nhưng không OCR lại ảnh đã hoàn thành hợp lệ.

## Trán bia: kết quả và giới hạn đã xác nhận

- Lượt dọc: `ocr_id=1, reading_direction=1, lang_type=1, epitaph=1, tag=null`.
- Lượt ngang đang thử: `ocr_id=4, reading_direction=2, lang_type=1, epitaph=0, tag=null`.
  Lần thử ngang trước với `epitaph=1` cũng không giải quyết đủ trán.
- Hai lượt dùng cùng ảnh, cùng tọa độ; OCR dọc gốc giữ nguyên. Phần ghép lưu riêng.
- Chọn trán hiện dựa vào dải hình học phía trên cụm cột đứng, không phải model layout.
  Box tự chọn luôn cần duyệt; một box có cả chuỗi không được tự đảo chữ OCR.
- Bia 99: API ngang trả 0 box, không phải bộ lọc làm mất.
- Bia 17065: chỉ nhận một phần hàng trán, không được gọi là hoàn thành.
- Bia 17019: API có một box trên một chữ trán, bộ lọc cũ loại vì vượt mép dải tìm.
  Đã sửa tại `SELECTION_VERSION=1.1`: cho phép phần lấn giới hạn, tâm phải ở trên
  mép thân, ít nhất 60% chiều cao ở phía trên. Kiểm tra cache trong bộ nhớ giữ được
  một box đó; các chữ API chưa phát hiện vẫn thiếu. Chưa ghi lại output ảnh thật.
- Các mẫu khác cũng chưa lấy đầy đủ trán. Chưa chứng minh do màu chữ hay độ khó ảnh;
  không tiếp tục đổi tham số tùy ý hoặc coi tests giả lập là bằng chứng OCR đạt.
- API là mô hình phía máy chủ, chưa có source/trọng số/cấu hình backend để biết nguyên
  nhân nội bộ. Tài liệu API đề cập `tag=model_id` chuyên biệt, nhưng chưa được xác nhận model.

## Phát hiện mới nhất: ZIP giao diện có model cục bộ

ZIP đã được người dùng khôi phục ở `../tools_lab/web_ui_ocr.zip` so với repository.
Đã đọc trực tiếp trong ZIP, chưa giải nén, chạy model hoặc cài thêm thư viện.
Không đưa ZIP, môi trường đóng gói, tài liệu mẫu hay thông tin xác thực của công cụ lên Git.

- `model/layout/best.pt`: DocLayout-YOLO, metadata có `Title`, `Content`, `Notes`.
  `align/nom_process.py` có `LayoutDetector` tải qua `doclayout_yolo.YOLOv10`.
- `model/nom/best.pt`, `model/nom/best_v2.pt`: YOLO DetectionModel, nhãn paragraph.
- `model/vi/best.pt`: YOLO DetectionModel, dùng phát hiện/cắt vùng đoạn văn.
- Đây là model phân vùng/phát hiện, không phải trọng số nhận dạng chữ Hán Nôm.
  `nom_ocr/ocr_client.py` và `web_ui/ocr_processor.py` xác nhận OCR Hán Nôm gọi API.
- Hướng đáng kiểm tra tiếp: model layout có nhận được vùng trán/thân/ghi chú trên bia
  không. Chưa được thử; `Title` trong model không bảo đảm chính là trán văn bia.
- `.venv-ocr` hiện không còn PaddleOCR/PaddlePaddle; trước đây từng chạy PP-OCRv6
  cục bộ, nhưng môi trường/model thử đã được dọn theo yêu cầu người dùng.

## Kiểm tra và trạng thái dự án

- Bộ tests gần nhất: **163 tests PASS**, ảnh/PDF giả lập và API giả lập, không gọi mạng.
  Chạy: `.venv-image/Scripts/python.exe -B -m unittest discover -s tests -q`.
- Hình thật vẫn cần kiểm tra; chưa đạt yêu cầu trán đầy đủ.
- Giữ nguyên `main.py`: file có sẵn mang đuôi `.py` nhưng nội dung nhị phân,
  chưa được phép sửa. Entry point Thành viên 2 là `run_image_pipeline.py`.
- Repository được chuyển từ thư mục lồng ra root. Tracked text/dataset cũ chỉ đổi
  đường dẫn; dữ liệu Lab mới và kết quả xử lý không nằm trong commit.
- Repository có sparse checkout và partial clone: không disable sparse checkout
  hoặc fetch/materialize toàn bộ tài liệu ngoài nhu cầu.
