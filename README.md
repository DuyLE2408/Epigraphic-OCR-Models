# Epigraphic-OCR-Models

Repository layout: `src/`, `data/`, the requirements files, and scripts now sit
directly beside this README at the Git root. The former nested
`Epigraphic-OCR-Models/` application directory has been removed. All project
paths are relative to this single root unless stated otherwise.

For Member 2, select `.venv-image/Scripts/python.exe` and open
`run_image_pipeline.py`. Run/F5 now processes one raw PDF/JPEG or All:
native-resolution RGB/JPG preparation → Kim Han Nom school API → raw OCR JSON → local box filtering,
repair and reading order in 03_filter_sort. A masked dialog requests the website cookie token when needed;
credentials are never saved. Python's standard HTTP library is used, so no new
packages or local PaddleOCR models are required.

The default Run adds a full-image horizontal pass (ocr_id=4,
reading_direction=2, epitaph=0) to try to detect missing headers. Vertical OCR stays unchanged;
horizontal results live in `02_ocr/<image_id>/header/general/`, and
`ocr_combined.json` combines selected headers with body boxes on one image.
Stage 03 sorts header boxes right-to-left and marks automatic selections for review.
Selection uses a geometric search band above a dense body-column group; it can
miss or misclassify titles and needs a real-image trial. No cropped image is sent.
Existing vertical results are reused; no need to delete old outputs or progress.
A fresh two-pass image uses at least four API requests; adding the horizontal pass
to an existing vertical result uses two. Quota checkpoints apply to both passes.
The earlier horizontal epitaph=1 trial returned zero raw boxes on stele 99.
That attempt is preserved under `header/`; the new general-text trial is separate.
Whether this setting improves real detection has not been established.

OCR inputs are no longer resized to 1024 px. Compatible RGB JPEGs are copied
without recompression; PDFs use embedded extraction or a 300 DPI page render.
Boxes use pixels of the prepared image throughout stages 02–03. The 1024 px
dataset image limit applies at final export, with coordinates transformed then;
vertical patches will be cropped before rotation and height reduction to 96 px.
Crop and dataset export are still pending. Old resized processing results must
not be reused as native-resolution OCR results.

A shared work queue at `data/image_pipeline/school_api/progress.json` records
completed (through stage 03) and pending images. Existing OCR-only results are
resumed locally before processing new API work; no image is resent for stage 03.
All processes only pending entries; completed
images do not enter the processing loop. Use `--recheck-completed` for a full
integrity check after manually editing outputs. Quota exhaustion stops the run with a saved
report; the next Run continues pending stages. Reports are retained under
`data/image_pipeline/run_reports/<run_id>/`. Use `--normalize-only` for the
previous local-only normalization mode. Images use `<stele_id>.<image_index>.jpg`;
supplied pages remain optional metadata and full books are not searched.
Run/F5 also offers a separate local filter/repair/order mode for existing OCR results.
It removes invalid/outside/identical duplicate boxes, orders confirmed headers
right-to-left and body columns left-to-right/top-to-bottom. Unconfirmed regions
remain provisional. Pixel/geometry heuristics can merge aligned fragments and
extend short truncated columns. Uncertain extensions remain proposals; changed
boxes retain source provenance and flag OCR refresh. No extra API calls or packages
are needed. Real-image accuracy still needs validation; patches and alignment are pending.
Follow [the Image Pipeline guide](docs/image_pipeline.md) for outputs, quota
handling and the first real-image test.

Processing stages live directly under
`data/image_pipeline/<book_id>/<stele_id>/01_prepare`, `02_ocr/<image_id>` and
`03_filter_sort/<image_id>`.
No Member 3 handoff is generated yet; `data/dataset` will be handled later.
Existing dataset files remain unchanged.

PROJECT: BUILDING A TRAINING DATASET FOR INSCRIPTION OCR MODELS

    Institution: VNUHCM-University of Science
    
    Team Members: Lê Trần Khánh Duy - Huỳnh Đức Dương - Lê Quí Thịnh
    
    Instructors: Mr. Đinh Điền, Mr. Nguyễn Hồng Bửu Long and Mr. Lương An Vinh

## 1. Project Overview
This project focuses on collecting, preprocessing, and building a standardized dataset to train two AI models:

 a. Inscription OCR Model: Detection and recognition of Sino-Nom characters on ancient stone stele rubbings.

 b. Translation Alignment Model: Sentence-level alignment across three components: Sino-Nom characters – Sino-Vietnamese Transliteration – Vietnamese Translation.

Image data is retrieved from the open-access portal Vietnamica (gallery.vietnamica.online) and expert-annotated research materials provided by the CLC Lab.

## 2. Project Structure
The project is structured following standard open-source conventions for readability and maintainability:

Epigraphic-OCR-Models/
├── data/                       # Inputs, intermediate files, and per-book datasets
│   ├── raw/<book_id>/          # Original files directly in the book folder; IDs from filenames
│   ├── processed_text/         # Member 1 text data; existing organization is preserved
│   ├── image_pipeline/<book_id>/<stele_id>/
│   │   ├── 01_prepare/        # Native-resolution RGB/JPG, metadata and local comparison
│   │   ├── 02_ocr/            # Raw OCR, boxes, previews and checkpoints
│   │   └── 03_filter_sort/    # Local filtering, automatic repairs, reading order and previews
│   └── dataset/
│       └── <book_id>/          # One complete dataset per source book
│           ├── images/        # <stele_id>.<image_index>.jpg, longest edge <= 1024px
│           ├── patches/       # Future: stele code + box ID, height <= 96px
│           ├── det_gt.txt     # Detection labels exported by Member 3
│           ├── rec_gt.txt     # Recognition labels with verified Sino-Nom text
│           └── results.xlsx   # Per-book output workbook
│
├── src/                        # Core source code
│   ├── config.py               # Global configurations (Paths, image resize thresholds, etc.)
│   ├── text_processing.py      # Raw text extraction & cleaning module (Member 1)
│   ├── image_ocr.py            # PDF/JPEG loading and normalization (Member 2)
│   ├── image_batch.py          # Batch normalization, validation and per-file reports
│   ├── school_ocr.py           # School API, resumable OCR and box previews (Member 2)
│   ├── header_ocr.py           # Full-image horizontal pass, header proposals and combined boxes (Member 2)
│   ├── box_postprocess.py      # Conservative filtering and configurable reading order
│   ├── box_repair.py           # Local fragment merging and bounded column extension
│   ├── image_paths.py          # Per-stele stage and handoff paths
│   ├── page_reference.py       # Supplied page metadata and image-status report (Member 2)
│   ├── alignment.py            # BertAlign / Levenshtein alignment algorithm (Member 3)
│   └── utils.py                # Helper functions for I/O operations and formatting
│
├── .gitignore                  # Git ignore rules for large data files
├── run_image_pipeline.py       # Current Member 2 Run/F5 entry point
├── requirements.txt            # Python dependencies
├── main.py                     # Entry point executing the full end-to-end pipeline
└── README.md                   # Project documentation

The assignments currently cover three source books: `di_san_han_nom_huyen_dai_loc`
(17 stele codes), `van_bia_thoi_tran` (7 texts, 12 faces), and
`van_khac_han_nom_thang_long_ha_noi`. The supplemental workbook currently displays
STT 80-123, including 28 image-bearing entries/30 faces; hidden rows are audited
separately. All three books have dataset
scaffolding; empty TXT/workbook files are not completed annotations. The one
existing pilot remains named `bia_20367.001.jpg` inside the Dai Loc book's
`images/` directory. New names use the stele ID and image index, independently
of the optional book page.
Per-book manifests live in `data/image_pipeline/<book_id>/manifest.json`.
See the Image Pipeline guide for the current layout and updated `--book-id` CLI.
New runs retain supplied page metadata only; missing pages do not block processing.
Existing outputs remain unchanged.

Raw inputs use `data/raw/<book_id>/<original_filename>`. The first numeric code
in each filename identifies its stele; original names and bytes are preserved.
Unresolved assignments are preserved under `data/raw/<book_id>/_can_doi_chieu/`.
The local inventory report is `data/image_pipeline/input_inventory/bao_cao_kiem_ke.md`;
it distinguishes missing references, combined images, and visible/hidden scopes.

## Coding & Naming Conventions

# Member 1: File src/text_processing.py

def extract_text_from_pdf(pdf_path: str) -> str:
    """OCR file PDF tài liệu thô thành chuỗi văn bản."""
    pass

def clean_raw_text(raw_text: str) -> str:
    """Loại bỏ chú thích, ký tự thừa, hình ảnh nhiễu."""
    pass

def split_text_types(cleaned_text: str) -> dict:
    """
    Tách văn bản làm sạch thành 3 luồng riêng biệt.
    Returns:
        dict: {
            "han_text": list[str],        # Danh sách các câu chữ Hán
            "phien_am_text": list[str],   # Danh sách các câu Phiên âm
            "dich_nghia_text": list[str]  # Danh sách các câu Dịch nghĩa
        }
    """
    pass

# Member 2: File src/image_ocr.py

import numpy as np

def resize_page_image(image_path: str, max_dim: int = 1024) -> np.ndarray:
    """Resize ảnh trang bia sao cho cạnh lớn nhất <= 1024px."""
    pass

def run_paddle_ocr(image: np.ndarray) -> list[dict]:
    """
    Chạy PaddleOCR để lấy bounding box và chữ Hán OCR.
    Returns:
        list[dict]: [
            {
                "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "sino_nom_ocr": "文字"
            }, ...
        ]
    """
    pass

def process_and_crop_patch(image: np.ndarray, bbox: list, patch_id: str) -> np.ndarray:
    """
    Cắt cột ảnh (patch), xoay 90 độ ngược chiều kim đồng hồ (nếu đứng), 
    resize height <= 96px.
    """
    pass

def sort_tran_bia_boxes(boxes: list) -> list:
    """Đảo thứ tự đọc chữ ở trán bia từ Phải sang Trái."""
    pass

# Member 3: File src/alignment.py and src/exporter.py

- src/alignment.py:

def align_ocr_with_ground_truth(ocr_boxes: list[dict], han_text_list: list[str]) -> list[dict]:
    """Gióng hàng chữ OCR từ ảnh với bản chữ Hán chuẩn (Dataset 1)."""
    pass

def align_translation_sentences(han_list: list[str], phien_am_list: list[str], dich_list: list[str]) -> list[dict]:
    """Gióng hàng theo cặp câu 3 thành phần (Dataset 2)."""
    pass
    
- src/exporter.py: 

def export_det_gt(aligned_data: list, output_file: str = "det_gt.txt") -> None:
    """Xuất file det_gt.txt theo đúng định dạng JSON points."""
    pass

def export_rec_gt(aligned_data: list, output_file: str = "rec_gt.txt") -> None:
    """Xuất file rec_gt.txt (đường dẫn patch \t chữ Hán)."""
    pass

def export_results_excel(aligned_data: list, output_file: str = "results.xlsx") -> None:
    """Xuất file results.xlsx chứa các cột Image ID, Patch ID, Bounding Box, SinoNom OCR."""
    pass

## 3. Pipeline & Workflow
The data preparation pipeline consists of 4 main stages:

graph TD
    A[Raw PDF/Image Documents] -->|Google Vision| B(Extracted Text: Sino-Nom, Transliteration, Translation)
    C[Vietnamica Stele Images] -->|PaddleOCR V6| D(Bounding Boxes & Cropped Patches)
    B --> E[Alignment Engine: BertAlign / Levenshtein]
    D --> E
    E --> F[OCR Dataset: det_gt.txt, rec_gt.txt, results.xlsx]
    E --> G[Translation Dataset: Sino-Nom - Transliteration - Translation]

a. Stage 1 (Raw Text Processing): Perform OCR on research documents to separate text into 3 distinct files (Sino-Nom, Transliteration, Vietnamese Translation) and clean up noise/annotations.

b. Stage 2 (Image Processing & PaddleOCR):
    - Resize main page images so the longest side <= 1024 pixels.
    - Run PaddleOCR V6 to extract bounding boxes. Filter out rubric/provenance metadata boxes.
    - Re-order stele headers (reading right-to-left).
    - Crop vertical text lines (patches), rotate 90° counter-clockwise, and scale height to <= 96 pixels.

c. Stage 3 (Data Alignment): Apply alignment algorithms (BertAlign/Levenshtein) to map bounding box OCR results directly to ground-truth text.

d. Stage 4 (Dataset Export & Reporting): Generate target annotations (det_gt.txt, rec_gt.txt, results.xlsx) and compile summary statistics.

## 4. Installation & Usage
Prerequisites
- Python >= 3.8
- OpenCV, Pandas, PaddleOCR, BertAlign / Levenshtein

Setup Instructions
a. Clone the repository:

git clone https://github.com/[your-username]/Epigraphic-OCR-Models.git
cd Epigraphic-OCR-Models

b. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate        # On Linux/macOS
  or: venv\Scripts\activate     # On Windows

c. Install dependencies:
pip install -r requirements.txt

d. Execution:
- Run the full pipeline:
    python main.py
- Run an individual component (e.g., Image Processing):
    python src/image_ocr.py

## 5. Final Deliverables
The final submission folder sent to the Lab/Instructors includes:

 a. 3 Text Folders: Cleaned Sino-Nom, Transliteration, and Translation text files for each stele.

 b. OCR Dataset: Standardized directory structure containing images/, patches/, det_gt.txt, rec_gt.txt, and results.xlsx.

 c. Translation Dataset: Aligned triplets of Sino-Nom – Transliteration – Translation sentences.

 d. Cropped Patches Folder: Preprocessed line crops for training recognition models.

 e. Summary Report: Quantitative metrics on the total number of bounding boxes and aligned sentence pairs.

## 6. Team Roles & Contribution Breakdown

- Member 1: Lê Trần Khánh Duy - Text Pipeline (29%): Responsible for raw PDF OCR via Google Vision, text filtering, and organizing the 3 text ground-truth datasets.

- Member 2: Huỳnh Đức Dương - Image & Detection Pipeline (36%): Responsible for image acquisition/scaling, running PaddleOCR V6, manual bounding box corrections, stele header re-ordering, and patch extraction.

- Member 3: Lê Quí Thịnh - Alignment, Base Infrastructure & Export (35%): Responsible for repository setup/coding conventions, alignment algorithm integration, automated dataset generation, and final metric reporting.

Data Confidentiality Notice: All materials provided by CLC Lab are restricted to internal academic use only. Redistribution or sharing outside the institutional domain is strictly prohibited.
