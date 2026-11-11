# Image Pipeline — Thành viên 2

## Phạm vi hiện tại

Chọn một PDF một trang/JPG/JPEG hoặc chạy All cho cả ba sách:
đọc raw → chuẩn bị JPG/RGB **giữ nguyên kích thước** → gọi API Kim Hán Nôm
→ lưu OCR gốc → lọc/tự sửa/sắp box → lưu JSON và ảnh xem tại 03_filter_sort.

Không dò mã bia, OCR số trang, nội suy số trang hoặc đọc PDF sách đầy đủ.
Có chế độ riêng lọc/tự sửa/sắp box đã OCR tại 03_filter_sort, không gọi API.
Ba khối xử lý chính: 01_prepare → 02_ocr → 03_filter_sort.
Khối 03 gồm box_postprocess.py (lọc/sắp) và box_repair.py (tự sửa bằng ảnh/tọa độ).
Có lượt OCR ngang toàn ảnh để đề xuất box trán bia; chưa tạo patches hoặc alignment.
OCR thành công không đồng nghĩa đã tìm đủ cột hoặc đọc đúng chữ.
Người phụ trách tự chạy ảnh thật; tests dùng ảnh/PDF giả lập và API giả lập.

Không resize trước OCR. JPG RGB không cần chỉnh EXIF được sao chép nguyên byte;
ảnh cần chỉnh hướng/chuyển RGB được lưu JPG chất lượng 95, không giảm kích thước.
PDF một ảnh phủ trang được trích ảnh nhúng khi đủ điều kiện; PDF khác render
300 DPI (có thể chọn --render-dpi). Độ phân giải PDF render phụ thuộc DPI,
không thể gọi là độ phân giải gốc của mọi thành phần trong PDF.

Tọa độ bbox trong 02/03 là pixel trên ảnh 01_prepare sau xử lý hướng EXIF.
JSON ghi image_size, coordinate_space=prepared_image_pixels; metadata 01 có
source_image_size và processing.resolution_policy=native, max_dim=null,
resize_scale x=y=1. Hai trường image_size_before_resize/after_resize được giữ
để tương thích và bằng nhau trong luồng mới. Không còn cờ --max-dim trên launcher.
Ảnh preview giữ cùng kích thước, nét khung và số được tăng theo độ phân giải.

Hướng dẫn mô tả images là trang sách với cạnh lớn nhất 1024 px; không bắt buộc
thu ảnh bia trước OCR. Khi làm bước crop/export sau này: crop từ 01_prepare,
xoay cột đứng 90° ngược kim đồng hồ rồi thu chiều cao patch về tối đa 96 px.
Nếu ảnh toàn trang xuất dataset giảm còn tối đa 1024 px, phải đổi cả bốn điểm
bbox theo tỷ lệ sx/sy thực tế, giữ tọa độ gốc riêng. Các bước crop/export chưa có.

API nhận ảnh lớn đến mức nào hoặc có thu nhỏ bên máy chủ hay không vẫn cần
thử ảnh thật. Chương trình không tự giảm chất lượng khi gặp lỗi; HTTP 413 sẽ
dừng và giữ tiến độ. Ảnh đầy đủ độ phân giải không bảo đảm OCR tìm đủ chữ.

## Cài môi trường lần đầu

Dùng Python 3.13 64-bit đã được kiểm thử. Tại thư mục repository:

~~~powershell
py -3.13 -m venv .venv-image
.\.venv-image\Scripts\python.exe -m pip install -r requirements-image.txt
~~~

Đặt PDF/JPG nhận qua kênh được Lab cho phép vào data/raw/<book_id>/.
Môi trường ảo và ảnh Lab không được đưa lên Git; máy mới tự tạo môi trường.

## Chạy bằng nút Run

1. Mở Git root (nơi có README.md và run_image_pipeline.py) trong VS Code.
2. Chọn Python interpreter .venv-image/Scripts/python.exe.
3. Mở run_image_pipeline.py → Run Python File in Terminal,
   hoặc F5 → cấu hình “Raw → OCR trường → xử lý box (1 hoặc All)”.
4. Chọn **Chọn 1 ảnh** hoặc **All — toàn bộ 3 sách**.
5. Với một ảnh: chọn file trực tiếp trong data/raw/<book_id>/,
   kiểm tra mã sách/mã bia điền sẵn rồi nhấn OK.
   Với All: chương trình tự duyệt ba sách, in tiến độ từng file trong Terminal,
   không mở hộp thoại chọn từng ảnh; cuối cùng hiện tổng kết và nơi lưu báo cáo.
6. Khi có ảnh cần gửi API, dán **chỉ Value của cookie token** vào hộp thoại nhập ẩn.
   Đăng nhập tại https://kimhannom.fit.hcmus.edu.vn/ rồi lấy cookie bằng F12.
   Nếu Terminal đã có KIMHANNOM_TOKEN thì dùng giá trị đó, không hỏi lại.
7. Mở đường dẫn bbox_preview.png trong kết quả hoặc report.csv để kiểm tra khung.

Token chỉ dùng trong bộ nhớ phiên chạy, không ghi file hay báo cáo. Nếu token cũ
trong Terminal hết hạn, lấy token mới; có thể xóa biến cũ bằng
Remove-Item Env:KIMHANNOM_TOKEN -ErrorAction SilentlyContinue rồi Run để nhập lại.
Không gửi mật khẩu/token vào chat, Git hoặc ảnh chụp màn hình.

Không có hộp thoại nhập số trang. Cancel ở bước chọn sẽ thoát.
File không bắt đầu bằng mã bia (ví dụ STT4_...pdf của sách đầy đủ) và file thuộc
_can_doi_chieu bị chặn. Không cần tạo thư mục riêng cho từng bia.
Nguồn và output cũ không bị ghi đè.

## Thử OCR bổ sung trán bia

Run/F5 và CLI thông thường tự chạy hai lượt trên cùng ảnh 01_prepare:
OCR dọc cho thân bia, OCR ngang (ocr_id=4, reading_direction=2, lang_type=1,
epitaph=0: ngang thông thường) để thử tìm trán. Không crop, che, xoay hoặc thu nhỏ ảnh gửi API.
Ảnh đầu vào vẫn là một file; các box dùng chung tọa độ pixel của file đó.
Đây là kỹ thuật bổ sung để thử nghiệm, không phải yêu cầu OCR hai lượt của hướng dẫn.

Lần thử trước với ocr_id=4, reading_direction=2, epitaph=1 trên bia 99 trả về
0 box ngay ở API, không phải do bộ lọc loại mất. Cấu hình mới thử epitaph=0
để tách chế độ chữ ngang thông thường khỏi chế độ văn bia. Chưa xác định
epitaph=1 là nguyên nhân; cần chạy ảnh thật để so sánh. Kết quả mới lưu riêng
header/general/, giữ nguyên lần thử cũ tại header/. Không tự sửa box/điền chữ
để làm như OCR đã nhận ra tiêu đề. Nếu API vẫn trả 0 box, đổi ngưỡng chọn box
không thể khôi phục tiêu đề; cần hỏi phía trường về cấu hình/model phù hợp.

Với bia đã OCR dọc, chọn lại chính file raw trong chế độ Chọn 1 ảnh rồi Run.
Không xóa 01_prepare, 02_ocr hoặc progress.json. Kết quả dọc hợp lệ được dùng lại,
chỉ upload/OCR ngang bổ sung; kết quả 03 cũ được cất vào _history khi tính lại.
All cũng nhận biết ảnh còn thiếu lượt ngang và đưa về hàng đợi. Nên thử bia 99 trước.

Code tìm cụm ít nhất ba cột đứng gần nhau từ lượt dọc, ước lượng một dải tìm kiếm
phía trên và lấy box lượt ngang trong dải, kích thước phù hợp
với chữ tiêu đề và có chữ Hán. Các ngưỡng hình học là heuristic, không hiểu nội
dung chữ và không bảo đảm nhận đúng trán trên mọi bia. Chữ nhỏ, trán lẫn thân,
bia nghiêng hoặc thiếu nhiều cột có thể chưa chọn được. Toàn bộ lượt ngang vẫn lưu
để xem những box bị bỏ khỏi bộ ghép và lý do tại header_selection.json.

Từ selection_version=1.1, dải tự ước lượng cho phép box lấn nhẹ xuống dưới mép
trên thân: tâm vẫn phải ở phía trên, ít nhất 60% chiều cao box ở phía trên và
phần lấn không quá một độ rộng cột thân. Không cắt bớt polygon để ép box vừa dải.
Vùng do người dùng khai báo vẫn yêu cầu chứa hoàn toàn. Sửa này giữ được một
box trán đã có trong cache bia 17019; không tạo được các chữ API chưa phát hiện.

Box được chọn có region=header, region_source=horizontal_header_candidate,
source_ocr_pass và source_ocr_index để truy ngược nguồn. Bước 03 đặt chúng trước
thân, sắp phải sang trái theo hàng và ghi automatic_header_needs_review.
Không tự đảo chuỗi sino_nom_ocr trong box chứa nhiều chữ; trường hợp này có cờ
header_text_direction_needs_review. Không suy ra nội dung chữ từ vị trí box.
Các box dọc trùng vị trí tiêu đề được thay trong bộ ghép, vẫn giữ nguyên ở OCR dọc.
Box thân còn lại giữ số ocr_index cũ; box từ lượt ngang được cấp số sau số dọc lớn nhất.
Sau sắp, patch_id/reading_order mới vẫn liền nhau. Không thay phần ghép/kéo box thân bia.

Trong preview kết quả ghép/bước 03, màu cam là box trán bia đề xuất hoặc đã chỉ định.
Cần xem lại; màu cam không có nghĩa đã xác nhận đúng. header_status:

- candidates_found: có box đề xuất; vẫn phải kiểm tra trán có đủ và đúng không.
- not_found: có vùng tìm nhưng không chọn được box; không kết luận bia vô đề.
- region_unknown: chưa đủ cột để ước lượng vùng trán; không tự nhận vùng trên ảnh.

Cấu hình vùng tùy chọn, chỉ khi cần sửa cách chọn sau khi xem thử:
02_ocr/<image_id>/header_regions.json chứa {"regions": [[x1,y1,x2,y2], ...]}.
Tọa độ theo ảnh 01_prepare, áp dụng riêng một ảnh. null dùng cách tự đề xuất.
Đây chỉ là vùng lọc kết quả trên máy, ảnh gửi lên vẫn đầy đủ. Đổi vùng rồi Run
sẽ chọn lại từ cache, không gọi API lại. Quy tắc này khác header_regions ở
03/rules.json: vùng 03 chỉ phân loại box đã có, không gọi OCR hoặc tạo box mới.
Nếu khai báo header_present=false ở bước 03, các box trán tự đề xuất sẽ bị loại.

## Bước 03 — lọc, tự sửa và sắp thứ tự box đã OCR

Bước này chạy cục bộ trên kết quả 02_ocr, không cần token và không gọi API.
Luồng Run mặc định tự chạy đến 03_filter_sort cho từng ảnh. Chế độ riêng này
dùng khi chỉ muốn xử lý lại box đã OCR, không chạy luồng raw/API.
Nên thử một ảnh trước khi chạy All.

1. Mở run_image_pipeline.py → Run → chọn **Lọc / tự sửa / sắp box đã OCR (không gọi API)**.
   Hoặc F5 → cấu hình **Lọc / tự sửa / sắp box đã OCR (không API)**.
2. Chọn **Chọn 1 ảnh** và chọn file nguồn trong raw; xác nhận mã sách/bia điền sẵn.
3. Chương trình tự lọc → phân tích/ghép/mở rộng box đủ điều kiện → sắp và đánh số.
   Chế độ mặc định không hỏi số box. Không gọi OCR lại trong bước này.
4. Mở bbox_preview.png ở 03_filter_sort: màu cam là trán bia đề xuất/đã chỉ định,
   xanh là box còn lại, tím là box đã tự sửa. Đây chỉ là box thực tế sau xử lý.
   Đề xuất chưa áp dụng được vẽ màu vàng trong proposals_preview.png riêng;
   không xuất hiện trong ảnh kết quả chính. Xem report.json để biết lý do.

Nếu cần khai báo vùng: chọn **Chọn 1 ảnh + khai báo vùng (tùy chọn)** thay cho
Chọn 1 ảnh. Nhập số box trán bia và box cần loại theo bbox_combined_preview.png
ở 02_ocr (hoặc bbox_preview.png nếu chưa ghép hai lượt), ví dụ
1,2,5-7; 0 nghĩa là đã kiểm tra không có, để trống giữ cấu hình trước đó/chưa rõ.
Cancel sẽ dừng. Không dùng số đã đánh lại ở 03_filter_sort để khai báo vùng.

All chỉ xử lý các JSON OCR đã có, không xử lý ảnh chưa OCR và không dùng chung
số box trán bia/loại bỏ cho tất cả ảnh. Nó dùng rules.json riêng từng ảnh;
nếu chưa có cấu hình vùng, kết quả là provisional (thứ tự tạm).

~~~powershell
.\.venv-image\Scripts\python.exe -B .\run_image_pipeline.py --postprocess-only --all
~~~

Quy tắc theo file hướng dẫn gốc:

- Trán bia: hàng trên trước, trong hàng phải sang trái.
- Phần còn lại: cột trái trước, trong cột trên xuống dưới.
- Bỏ ghi chú địa điểm rập bia khi đã xác định đúng vùng/box.
- Chỉ cần vẽ/chỉnh tọa độ khi có lỗi; module mới tự sửa một số lỗi có đủ bằng chứng.

Module box_repair.py dùng Pillow và thư viện chuẩn, không thêm OpenCV/model/API.
Nó ghép các đoạn box đứng có tâm/độ rộng gần nhau, khoảng cách ngắn, có tương phản
và nét ảnh trong cả hai đoạn, không vướng box khác hoặc vùng đã loại.
Cho phép cột nghiêng nhẹ đến độ trôi ngang/chiều cao 0,05; so vị trí các đoạn tại
khoảng nối thay vì so hình chữ nhật bao toàn cột. Ghép xong mới sắp và đánh số
liên tục từ 1; số API gốc vẫn giữ trong source_ocr_indices. Hai cột cùng phủ chiều
dọc không bị nhóm thành một cột chỉ vì hình chữ nhật của chúng chồng nhau theo x.
Sau đó phân tích mật độ nét theo hàng trong dải cột để tìm phần tiếp nối ở đầu/cuối.
Tự chọn nét sáng hoặc nét tối từ box nguồn, điều chỉnh ngưỡng theo vùng nhỏ ngoài box,
và lấy mẫu theo độ nghiêng; dừng khi gặp khoảng trống hoặc vạch ngang đặc.
Phần mở rộng phải có cấu trúc nét tương tự nhiều ô chữ trong cột gốc. Một ô riêng
lẻ chỉ được chấp nhận khi có thêm hai cột lân cận xác nhận vị trí cuối/đầu gần đó.
Chỉ kiểm tra va chạm trong phần mới mở rộng, không coi độ chồng có sẵn giữa các
box OCR hơi nghiêng là lỗi mới. Mở rộng giữ tứ giác theo độ nghiêng của box.
Chỉ tự áp dụng mở rộng ngắn, tối đa 2,5 lần độ rộng cột, trong phạm vi thân bia
ước lượng từ các box có sẵn. Có thể tới 5 lần độ rộng khi nét ảnh có bằng chứng
và ít nhất hai cột lân cận xác nhận điểm đầu/cuối tương ứng. Không kéo tất cả
cột về cùng một đáy chỉ vì chúng đứng cạnh nhau. Mở rộng yếu/vướng box khác chỉ lưu đề xuất.
Các ô nét kéo thêm phải có mật độ và độ phức tạp đủ gần với ô chữ trong box gốc;
vài nét trắng riêng lẻ không tự được coi là thêm chữ.
Vùng tìm đầu/cuối giới hạn 6 lần độ rộng cột; đề xuất chạm giới hạn này chưa
khẳng định đã tìm hết phần chữ còn thiếu và không được tự áp dụng.
Box nghiêng nhiều, box tiêu đề, ảnh ít tương phản hoặc thiếu bằng chứng được giữ nguyên.

Sau sửa hình học, bước tighten_sides thu khoảng trống bên hông nếu có bằng chứng:
giới hạn mỗi bên 20% độ rộng, có đệm bảo vệ 1,5 px, giữ ít nhất 65% độ rộng cũ
và ít nhất 99% foreground đã phân đoạn. Đây là bảo vệ pixel, không chứng minh
mọi nét chữ đều được nhận diện đúng; ảnh nhiễu có thể không thu được hoặc chỉ thu rất ít.
Không tự cắt đầu/cuối box để bỏ một ký tự nghi vấn. Tắt riêng thu mép bằng
tighten_boxes=false trong rules.json, vẫn giữ ghép/mở rộng và kiểm tra đuôi.

Đây là heuristic cần kiểm tra, không phải mô hình hiểu chữ: vẫn có thể nhầm
hoa văn với nét chữ hoặc ghép hai đoạn văn bản độc lập. Đã đối chiếu một trường hợp
ảnh thực trong bộ nhớ; chưa đánh giá độ chính xác trên toàn bộ tập ảnh.
Khối sửa box không tự tìm cột bị bỏ sót hoàn toàn. Lượt OCR ngang riêng chỉ đề xuất
trán bia theo hình học, chưa tự xác nhận đúng tiêu đề hoặc phân loại ghi chú.
Thân bia có bố cục nhiều vùng, chữ nghiêng hoặc xen kẽ vẫn cần kiểm tra thứ tự.
Một box trán bia chứa cả chuỗi ngang nhiều chữ được cảnh báo cần kiểm tra
chiều chuỗi; đảo vị trí các box không giải quyết được thứ tự bên trong một box.

Tự loại chỉ gồm tứ giác không hợp lệ, box hoàn toàn ngoài ảnh, box trùng chính xác
cả tọa độ lẫn chữ OCR, hoặc box được chỉ định loại qua cấu hình. Box nhỏ/độ tin cậy
thấp vẫn giữ lại; box chỉ một phần ngoài ảnh vẫn giữ và cảnh báo. Không dùng độ
tin cậy để suy ra box thừa. Chương trình chưa thể tự khẳng định đâu là trán bia
hay ghi chú từ JSON API hiện tại. Có thể tắt sửa tự động bằng auto_repair=false
trong rules.json của riêng ảnh; lọc và sắp vẫn hoạt động.

~~~text
data/image_pipeline/<book_id>/<stele_id>/03_filter_sort/<image_id>/
├── rules.json          # Quy tắc của riêng ảnh; số box gốc, vùng và trạng thái xác nhận
├── boxes.json          # Box còn lại đã sắp; giữ liên kết về OCR nguồn
├── bbox_preview.png    # Ảnh xem thứ tự mới, không sửa ảnh 01_prepare
├── proposals_preview.png # Đề xuất màu vàng chưa áp dụng, tách khỏi ảnh kết quả chính
├── report.json         # Box giữ/loại, repair.changes, nguyên nhân và cảnh báo
├── complete.json       # Checksum nguồn, quy tắc và kết quả
└── _history/           # Có khi đổi quy tắc/nguồn: giữ kết quả sinh trước đó
~~~

boxes.json lưu image_id, book_id, đường dẫn/checksum OCR nguồn, rules, warnings,
order_status và review_required. Box chưa sửa giữ bbox và sino_nom_ocr gốc;
source_patch_id/ocr_index truy lại box API; patch_id mới gồm image_id và số
thứ tự ba chữ số, reading_order bắt đầu từ 1, reading_group và region.
patch_path vẫn null, sino_nom chưa được gióng hàng. configured chỉ có nghĩa
đã cung cấp quy tắc vùng, không phải xác nhận OCR hay detection chính xác.
review_required luôn true trước bước kiểm tra/crop.

Box đã sửa lưu source_boxes với tọa độ/chữ/ID từng box gốc, source_ocr_indices,
source_patch_ids và ocr_needs_refresh=true. Ghép box nối các chuỗi OCR gốc theo
trên xuống dưới, ghi joined_fragments_unverified. Mở rộng box giữ chữ OCR cũ và
ghi partial_after_box_extension: phần chữ mới chưa được nhận diện, không tự sinh chữ.
Module không tự gọi API để OCR lại. sino_nom vẫn null, không coi OCR là ground truth.
repair.changes ghi bbox trước/sau, lý do và applied=true/false; unassessed ghi
những cột không đủ bằng chứng ảnh để đánh giá. Báo cáo tổng hợp thêm số lần tự sửa,
đề xuất và box cần OCR lại.
repair.tail_reviews lưu các vùng cuối cột có nét yếu, thiếu độ phủ hoặc hoạt tính,
kèm reading_order hiện tại, bbox vùng cần xem và requires_text_alignment=true.
Đây là yêu cầu đối chiếu, không phải kết luận nét đó không phải chữ. Không tự xóa
hay thay ký tự OCR trong những vùng này. Số vùng được ghi ở tail_review_count,
CSV, tổng kết và Terminal. Cần đối chiếu tài liệu chuẩn ở bước alignment.

Chạy lại kiểm tra checksum: kết quả hợp lệ được giữ nguyên; đổi quy tắc/phiên bản thuật toán thì
cất kết quả sinh cũ vào _history rồi tính lại. Nếu kết quả bị sửa/hỏng thủ công,
chương trình báo lỗi và giữ file. Không sửa JSON/ảnh của 02_ocr.
Để đổi quy tắc, chọn khai báo vùng tùy chọn hoặc chỉnh riêng rules.json rồi chạy lại;
vùng tùy chọn header_regions/exclude_regions dùng [x1,y1,x2,y2] trên ảnh 01_prepare,
chỉ áp dụng khi box nằm hoàn toàn trong vùng. Vùng giao nhau cần kiểm tra kỹ.

Trạng thái bước này lưu tại data/image_pipeline/filter_sort/progress.json;
chạy All kiểm tra cache từng kết quả OCR đã có, tiếp tục tính những kết quả thiếu.
File này độc lập với hàng đợi school_api/progress.json và không thay đổi hạn mức API.
Báo cáo mỗi lần chạy có report.json, report.csv và summary.md trong run_reports.

## Lưu tiến độ và dừng khi hết hạn mức API

Mỗi ảnh được xử lý tuần tự: chuẩn bị → upload/OCR dọc → upload/OCR ngang
→ ghép kết quả → lọc/sắp box. Các lượt đã có checkpoint hợp lệ được dùng lại.
Không gửi nhiều ảnh cùng lúc. Mỗi request có User-Agent và Bearer token theo
PDF API Kim Hán Nôm phiên bản 10/10/2026. HTTPS được kiểm tra chứng chỉ;
không chuyển tiếp yêu cầu sang miền khác khi gặp redirect.

Lượt dọc: ocr_id=1 (thông thường dọc), lang_type=1 (Hán),
reading_direction=1 (dọc), epitaph=1 (văn bia), tag=null (chưa cấp model_id riêng).
Lượt ngang: ocr_id=4, reading_direction=2, epitaph=0; Hán và tag giữ nguyên.
Không cần gọi API phân loại vì đã biết loại tài liệu.

Chương trình áp dụng bảo thủ các hạn mức website ghi trong PDF:
25 lượt/phút, 50 lượt/giờ, 200 lượt/24 giờ và 600 lượt/7 ngày.
Các cửa sổ được tính liên tục bằng lịch sử request cục bộ, cách nhau ít nhất 3,1 giây.
Một ảnh mới chạy đủ hai hướng cần ít nhất bốn request: hai upload và hai OCR.
Ảnh đã có lượt dọc cần thêm hai request cho lượt ngang; ảnh đã xong cả hai không gửi lại.

- Chạm hạn mức: lưu báo cáo trạng thái paused, thời điểm thử tiếp và dừng.
- HTTP 429: dùng Retry-After nếu có; nếu thiếu, tạm dừng một giờ rồi cho thử lại.
- HTTP 401/403: lưu tiến độ, dừng và yêu cầu kiểm tra token/quyền tài khoản.
- Lỗi kết nối/timeout: dừng, không tự lặp POST vì chưa biết máy chủ đã xử lý chưa.
  Lần Run sau có thể phát sinh lượt OCR mới nếu response trước đó chưa được lưu.
- Ảnh hỏng/lỗi riêng một ảnh: ghi nguyên nhân và tiếp tục ảnh sau.
- Ctrl+C: giữ các phần đã lưu, đánh dấu ảnh bị ngắt và ảnh chưa chạy.

Lần Run sau đọc hàng đợi chung tại data/image_pipeline/school_api/progress.json.
File này lưu target_stage=03_filter_sort, completed (ảnh đã xong cả ba bước), pending (ảnh chưa xong), next_source
(ảnh tiếp theo), last_status và thời điểm cập nhật. Tiến độ được ghi sau từng ảnh.
Ảnh hoàn thành có kích thước/thời gian sửa nguồn không đổi và dấu hoàn thành bước 3
còn khớp phiên bản/quy tắc và bộ ghép trán bia còn hợp lệ sẽ bị loại trước vòng xử lý.
Không mở ảnh bằng Pillow hay OCR lại ảnh đã hoàn thành. Có kiểm tra checksum
JSON và preview của hai lượt để tránh dùng bộ ghép sai/hỏng.
Chương trình vẫn đọc danh sách tên file để phát hiện ảnh mới thêm vào raw.

Với ảnh còn trong pending: chuẩn hóa hợp lệ thì dùng lại; upload đã lưu thì
chạy tiếp OCR; JSON gốc đã có thì chỉ dựng phần còn thiếu, không gọi API lại.
Ảnh có cache được ưu tiên làm bù; thiếu lượt ngang thì vẫn cần request bổ sung,
còn chỉ thiếu bước 3 thì chạy cục bộ.
Chỉ ghi completed sau khi bước 3 thành công. Nếu bước 3 lỗi, ảnh vẫn pending;
lần Run sau dùng lại OCR đã có để thử tiếp, không tự gửi ảnh lần nữa.
Ảnh làm dở hoặc lỗi vẫn nằm trong hàng đợi. Hạn mức API vẫn dùng request_history.json.

Lần đầu tạo progress.json sẽ kiểm tra các complete.json cũ một lần để nhập ảnh đã xong.
Nếu đã sửa/xóa output của ảnh hoàn thành, dùng chế độ kiểm tra lại:

~~~powershell
.\.venv-image\Scripts\python.exe -B run_image_pipeline.py --all --recheck-completed
~~~

Chọn riêng một ảnh cũng kiểm tra lại ảnh đó. Không tự sửa progress.json hoặc xóa
file này để chạy lại toàn bộ; giữ cache và lịch sử API. File tiến độ hỏng sẽ làm
chương trình dừng để kiểm tra, không âm thầm đặt lại hàng đợi.

Lịch sử lượt gọi nằm tại data/image_pipeline/school_api/request_history.json.
Không xóa file này để né hạn mức. Các lượt gọi trước đây, từ website hoặc chương trình
khác không được bộ đếm này biết; máy chủ vẫn có thể trả 429 sớm hơn.
Chỉ một phiên pipeline được chạy trong cùng repository tại một thời điểm.

## Kết quả OCR của mỗi ảnh

~~~text
data/image_pipeline/<book_id>/<stele_id>/02_ocr/<image_id>/
├── manifest.json       # Hash ảnh và tham số API, không có token
├── upload.json         # Tên ảnh tạm trên máy chủ để tiếp tục sau upload
├── ocr_raw.json        # Response thành công nguyên gốc, giữ nguyên chữ OCR
├── ocr.json            # Box bốn góc, ID và chữ OCR gắn với box
├── bbox_preview.png    # Ảnh cục bộ vẽ khung và số thứ tự API
├── complete.json       # Checksum lượt dọc
├── header/             # Giữ lượt thử cũ epitaph=1 nếu có
│   └── general/        # manifest/upload/ocr_raw/ocr/preview/complete của lượt ngang epitaph=0
├── ocr_combined.json   # Box dọc + box trán được chọn, dùng cho bước 03
├── bbox_combined_preview.png # Xem bộ ghép trên cùng ảnh nguồn; trán màu cam
├── header_selection.json # Vùng tìm, box được chọn/bỏ, trạng thái cần kiểm tra
├── combined_sources.json # Nguồn/checksum hai lượt; đây không phải response API
├── complete_combined.json # Checksum bộ ghép
└── _combined_history/  # Chỉ có khi đổi vùng/phiên bản cách chọn
~~~

ocr.json có image_id, book_id, image_path, image_sha256 và boxes.
Mỗi box có patch_id duy nhất theo image_id + số box ba chữ số, bbox, sino_nom_ocr,
confidence, ocr_index; reading_order, patch_path và sino_nom đang để null.
reading_order_status=not_reviewed và review_required=true nhắc rằng còn cần
kiểm tra box/thứ tự đọc. Chưa có patch thực tế hoặc nội dung đã alignment.
Không coi số thứ tự API là thứ tự đọc đã xác nhận.

Ảnh gốc trong raw và JPG chuẩn hóa không bị vẽ đè.
Các kết quả thử local_ocr_trial và school_api_trial cũ đã được dọn khi chuyển
sang giữ nguyên độ phân giải. Không dùng OCR trên ảnh nhỏ làm cache cho ảnh lớn.

## Cấu trúc xử lý của mỗi bia

~~~text
data/image_pipeline/<book_id>/<stele_id>/
├── 01_prepare/
│   ├── <image_id>.jpg
│   ├── <image_id>.json
│   ├── <image_id>.preview.html
├── 02_ocr/
│   └── <image_id>/
│       ├── manifest.json
│       ├── upload.json
│       ├── ocr_raw.json
│       ├── ocr.json
│       ├── bbox_preview.png
│       └── complete.json
└── 03_filter_sort/
    └── <image_id>/              # rules.json, boxes.json, report.json, preview, complete.json
~~~

Các thư mục giai đoạn nằm trực tiếp trong mã bia. Mã sách giúp tách những bia
cùng mã ở hai sách. Bước lọc/sắp thêm 03_filter_sort/<image_id> ngay dưới mã bia;
crop sẽ thực hiện sau.
Hiện chưa xuất dữ liệu bàn giao cho Thành viên 3; phần data/dataset sẽ làm sau.
Ảnh và nhãn đang có trong data/dataset được giữ nguyên.

Không dùng lại ảnh resize cũ cho luồng OCR giữ nguyên độ phân giải. Progress
có cấu hình khác sẽ bị chặn để tránh nhầm kết quả. Sau khi dọn output cũ theo
yêu cầu ngày 11/11/2026, progress được tạo lại khi Run; lịch sử hạn mức API,
kiểm kê đầu vào và số trang được cung cấp vẫn giữ nguyên. data/dataset còn
chứa kết quả cũ và chưa được cập nhật; chúng không chặn hay cấp số ảnh cho 01.

## Chạy bằng Terminal

Tại Git root:

~~~powershell
.\.venv-image\Scripts\python.exe -B run_image_pipeline.py --input "data/raw/di_san_han_nom_huyen_dai_loc/20368.pdf" --book-id di_san_han_nom_huyen_dai_loc --stele-id bia_20368 --preview
~~~

~~~powershell
.\.venv-image\Scripts\python.exe -B run_image_pipeline.py --input "data/raw/van_bia_thoi_tran/07804.jpg" --book-id van_bia_thoi_tran --stele-id bia_7804 --preview
~~~

Dùng --input cho cả PDF và JPG/JPEG.
Thêm --normalize-only nếu chỉ muốn chuẩn hóa, không gửi ảnh tới API trường.
Để mở giao diện lựa chọn cho chế độ này:

~~~powershell
.\.venv-image\Scripts\python.exe -B run_image_pipeline.py --normalize-only
~~~

--page-number (chỉ ở --normalize-only) ghi một trang tài liệu đã được cung cấp vào metadata.
Không nhập nếu chưa biết. Cờ này không đặt tên ảnh, không cấp trang tạm và không dò sách.
Thiếu trang hoặc thiếu Excel đều không ngăn chuẩn hóa ảnh.

Chạy toàn bộ bằng Terminal:

~~~powershell
.\.venv-image\Scripts\python.exe -B run_image_pipeline.py --all --preview
~~~

Không kết hợp --all với --input, --book-id, --stele-id, --output hoặc
--page-number. All xử lý đủ ba sách đã cấu hình. --render-dpi chọn DPI khi cần
render PDF; --max-dim đã bỏ để tránh thu nhỏ ảnh trước OCR.

## Báo cáo khi chạy All và quy tắc chạy lại

Mỗi lần pipeline tạo một thư mục riêng, giữ nguyên báo cáo các lần trước:

~~~text
data/image_pipeline/run_reports/<run_id>/
├── summary.md     # Tổng kết, thống kê từng sách và bảng lỗi dễ đọc
├── report.csv     # Một dòng cho mỗi file, mở bằng Excel
└── report.json    # Kết quả đầy đủ, thông số, lỗi, mục loại trừ và thời gian UTC
~~~

Báo cáo tách rõ tổng số ảnh/PDF nguồn, thành công mới, đã có hợp lệ nên bỏ qua,
lỗi, bị ngắt giữa chừng và chưa đến lượt. Từng file có tên/đường dẫn nguồn,
mã bia, trạng thái, nguyên nhân lỗi, loại lỗi và đường dẫn JPG/JSON khi có.
Lỗi một ảnh không ngăn các ảnh tiếp theo. Lỗi thư mục sách và các file/thư mục
không thuộc đầu vào cũng được ghi riêng, không âm thầm bỏ qua.

Ở chế độ OCR mặc định, progress.json loại ảnh đã hoàn thành khỏi vòng xử lý.
Khi dùng --recheck-completed hoặc --normalize-only, chỉ bỏ qua nếu checksum nguồn và output còn khớp JSON, JPG
đọc được/RGB/kích thước đúng và cấu hình chuẩn hóa không đổi. Không coi mọi lỗi
“file đã tồn tại” là hoàn thành. Nguồn đổi, thiếu/hỏng output, JSON hỏng hoặc
thông số khác được ghi lỗi cần kiểm tra; không ghi đè và không tự cấp số ảnh mới
để né một JPG/preview thiếu metadata.

Báo cáo được cập nhật sau từng file. Ctrl+C sẽ giữ kết quả đã xong, đánh dấu
file đang bị ngắt và các file chưa chạy; báo cáo không bị xóa. Nếu tắt cưỡng bức
Python/mất điện, báo cáo gần nhất có thể còn trạng thái running. Kiểm tra trước
khi chạy lại. CSV đang mở trong Excel có thể không cập nhật được; JSON vẫn là
bản ghi chính và chương trình sẽ báo cảnh báo ghi báo cáo.

## Tên ảnh và dữ liệu lưu

Dùng mã bia/nhóm bia và chỉ số ảnh, tách hoàn toàn khỏi trang tài liệu:

~~~text
data/raw/<book_id>/<file nguồn>
data/image_pipeline/<book_id>/<stele_id>/01_prepare/<stele_id>.001.jpg
data/image_pipeline/<book_id>/<stele_id>/01_prepare/<stele_id>.002.jpg
data/image_pipeline/<book_id>/<stele_id>/01_prepare/<stele_id>.001.json
data/image_pipeline/<book_id>/<stele_id>/01_prepare/<stele_id>.001.preview.html
data/image_pipeline/<book_id>/image_status.csv
~~~

Ví dụ bia_7804.001.jpg. 001 là ảnh thứ nhất của bia, không phải trang sách.
Mã bia lấy từ cụm chữ số đầu tên nguồn và bỏ số 0 đầu: 07804.jpg → bia_7804,
00396-22.jpg → bia_396. Với tên ghép bằng &, lấy mã đầu làm mã chính;
các mã phía sau không được tách thành ảnh mới. Bốn file đầu 04997, 04998,
04999, 05000 là bốn mã chính riêng. Tên và nội dung nguồn được giữ nguyên.
Mỗi mã bia/nhóm có chuỗi chỉ số ảnh riêng. JSON giữ ánh xạ đến file nguồn.
Chế độ một ảnh chặn ghi đè. All kiểm tra rồi bỏ qua kết quả hợp lệ đã có.
Nguồn đã có output tên sách–trang ở phiên bản trước cũng được giữ, báo cần kiểm tra.
Không tự đổi tên, xóa hoặc di chuyển output cũ.
Nếu nguồn trước đây nằm trong thư mục bia rồi được chuyển ra thư mục sách,
chương trình nhận lại qua tên file trong cùng sách; báo cáo kiểm tra checksum
để xác định nguồn còn nguyên hay đã đổi. Metadata cũ không tự sửa đường dẫn.

Các patches sau này dùng ID theo mã bia và thứ tự box theo hướng dẫn Dataset 1.
Chưa tạo patch hay ghi ground truth ở bước này.

## Số trang được cung cấp

src/page_reference.py chỉ đọc cột Trang có sẵn trong Excel và tạo báo cáo trạng thái.
Không còn worker Windows OCR, quét sách, nhận diện mã, chấm điểm trang hoặc nội suy.

- Thời Trần: đọc cột G (Trang), đối chiếu mã ở cột H của
  project_material/Nhom4/DanhSach.xlsx, tính từ thư mục NLP.
- Đại Lộc và Thăng Long Hà Nội: chưa có cột trang cung cấp cho từng ảnh, để null.
- Nhóm nhiều mặt giữ nguyên trang tham khảo của bài viết, cùng giá trị trên các mặt;
  page_scope=stele_entry ghi rõ phạm vi, không tự cộng trang cho từng mặt.
- Nhóm 4997–5000 áp dụng sửa mã đã được người phụ trách xác nhận; không sửa Excel gốc.
- Có thể truyền --page-number khi đã được cung cấp số trang rõ ràng.
- Excel thiếu/hỏng hoặc có nhiều dòng cung cấp trang xung đột: để trang trống, báo lý do.

Các page_reference.json và page_reference.csv cũ vẫn giữ nguyên làm lịch sử.
Pipeline không đọc lại các kết quả dò/OCR đó, không dùng số đã đoán để điền vào metadata mới.
Không cần tạo JSON tham chiếu để chạy ảnh.

## JSON chuẩn hóa schema 1.7

- image_id: mã bia/nhóm + chỉ số ảnh, ví dụ bia_7804.001.
- image_number: chuỗi ba chữ số, ví dụ 001.
- page_number và book_page_number: số nguyên được cung cấp, hoặc null.
- page_number_source: provided_excel, provided_explicit hoặc null.
- page_scope: stele_entry, user_provided hoặc null.
- filename_status: stele_image_index.
- book_reference: thông tin nguồn cung cấp trang; không có ứng viên OCR.
- source_path, source_type, source_sha256, source_unchanged: truy nguồn và checksum.
- output_jpg, output_jpg_sha256, metadata_path: đường dẫn và checksum kết quả.
- Kích thước trước/sau, cách đọc nguồn và thông số chuẩn hóa vẫn được giữ.

Không còn dataset_page_number giả làm trang. Chỉ số ảnh và trang tài liệu là hai trường riêng.
Metadata các phiên bản trước không bị sửa.
source_page/source_pdf_page=1 ở PDF là trang vật lý của file ảnh đầu vào một trang,
không phải trang trong sách. JPG không có trang PDF.

## Theo dõi ảnh đã xử lý

Mở data/image_pipeline/<book_id>/image_status.csv bằng Excel.
Bảng hiện tại gồm mã bia, file nguồn, trang được cung cấp, phạm vi/nguồn số trang,
trạng thái chuẩn hóa và đường dẫn output. Trang chưa biết là ô trống.

Báo cáo phân biệt chưa chuẩn hóa, đã chuẩn hóa, nguồn đã đổi, output thiếu/hỏng,
hoặc có nhiều output cần kiểm tra. Nó kiểm tra checksum nguồn và JPG theo metadata.
CSV chuẩn hóa cập nhật khi chạy --normalize-only. Luồng OCR mặc định dùng
progress.json và báo cáo từng lần chạy, không quét lại toàn bộ ảnh để cập nhật CSV này.
Đóng file trong Excel nếu cần chương trình cập nhật.
Bảng page_reference.csv cũ là lịch sử dò trang, không còn là báo cáo hiện hành.

## Chuẩn hóa và bảo toàn dữ liệu

- PDF: yêu cầu đúng một trang, không mã hóa hoặc cần repair. Ưu tiên trích ảnh nhúng
  khi đúng một ảnh độc lập, không mask/xoay/biến dạng, phủ trọn trang; còn lại render
  RGB nền trắng ở 300 DPI. Không chọn ảnh đầu tiên một cách mặc định.
- JPG/JPEG: kiểm tra định dạng JPEG thật, đọc đủ dữ liệu và tôn trọng hướng EXIF.
- Chuyển RGB; alpha ghép nền trắng; thu nhỏ LANCZOS, không phóng lớn, crop,
  đoán hướng, tăng tương phản, làm nét hoặc nhị phân hóa.
- Xuất JPEG quality 95, subsampling 0. Kiểm tra ảnh mã hóa và checksum nguồn trước/sau.
- Không ghi đè JPG/JSON/HTML đã có. Không sửa processed_text/, main.py,
  nhãn của Thành viên 3, dữ liệu ảnh gốc hoặc PDF sách.
- Chế độ --normalize-only không gọi API; chế độ mặc định gửi ảnh tới dịch vụ trường.
  Không tự cài thư viện, commit hoặc push.
- Manifest cũ là snapshot; chưa tự tổng hợp sau mỗi lần chạy.

## Môi trường và kiểm thử

Môi trường hiện có: Python 3.13 x64, Pillow 13.0.0, PyMuPDF 1.26.7.
Không cần Windows OCR hoặc dependency đọc Excel bổ sung; XLSX đọc bằng thư viện chuẩn.
Không cần cài lại môi trường nếu đã có .venv-image.
API trường dùng urllib có sẵn trong Python; Run/F5 không cần .venv-ocr,
requests hoặc mô hình PaddleOCR trên máy.

~~~powershell
.\.venv-image\Scripts\python.exe -B -m unittest discover -s tests -q
~~~

Tests dùng ảnh/PDF/XLSX giả lập để kiểm tra chuẩn hóa, EXIF, PDF extract/render,
file hỏng, nhiều trang, chặn ghi đè, tên theo mã bia, trang tùy chọn và báo cáo trạng thái.
Có kiểm tra Run không mở sách/khởi động OCR, không sử dụng bảng OCR cũ và không tự gán trang.
Tests All kiểm tra tiếp tục sau ảnh hỏng/PDF nhiều trang, báo cáo từng file,
bỏ qua kết quả hợp lệ, nguồn thay đổi, JPG thiếu JSON, giữ lịch sử và ngắt giữa chừng.
Tests sửa box dùng nét chữ giả lập sáng/tối, kiểm tra ghép đoạn, nối chuỗi theo
chiều dọc, mở rộng đầu/cuối, giữ cột bên cạnh, tránh vùng bị loại/vạch ngang,
đề xuất thay vì áp dụng sửa dài, lưu nguồn, cache và báo cáo mà không gọi HTTP.
Có kiểm tra box nghiêng nhẹ, đánh số liền sau ghép, phân biệt hai cột song song,
ảnh kết quả không chứa đề xuất vàng và nâng cấp cache cũ thiếu ảnh đề xuất riêng.

resize_page_image nhận đối số như README nhưng vẫn trả Pillow Image RGB.
Adapter numpy.ndarray/BGR, OCR trực tiếp trong image_ocr.py và hàm crop còn chờ triển khai.
sort_tran_bia_boxes(boxes) giữ chữ ký README; danh sách truyền vào được coi là
box trán bia, trừ box có region khai báo khác. Hàm trả bản sao đã sắp theo hàng,
phải sang trái, giữ nguyên chữ và tọa độ. Pipeline đầy đủ dùng box_postprocess.py
để phân biệt vùng trước khi sắp. Alignment vẫn dành cho giai đoạn sau.

Đầu vào PDF và JPG/JPEG dùng chung normalize_single_input; hàm bọc
normalize_single_pdf và cờ --pdf cũ đã được bỏ để tránh hai cách gọi trùng nhau.
