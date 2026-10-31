# Epigraphic-OCR-Models
# Epigraphic-OCR-Models
PROJECT: BUILDING A TRAINING DATASET FOR INSCRIPTION OCR MODELS
    Institution: VNUHCM-University of Science
    Team Members: Lê Trần Khánh Duy - Huỳnh Đức Dương - Lê Quí Thịnh
    Instructors: Mr. Đinh Điền, Mr. Nguyễn Hồng Bửu Long and Mr. Lương An Vinh

1. Project Overview
This project focuses on collecting, preprocessing, and building a standardized dataset to train two AI models:

 a. Inscription OCR Model: Detection and recognition of Sino-Nom characters on ancient stone stele rubbings.

 b. Translation Alignment Model: Sentence-level alignment across three components: Sino-Nom characters – Sino-Vietnamese Transliteration – Vietnamese Translation.

Image data is retrieved from the open-access portal Vietnamica (gallery.vietnamica.online) and expert-annotated research materials provided by the CLC Lab.

2. Project Structure
The project is structured following standard open-source conventions for readability and maintainability:

Epigraphic-OCR-Models/
├── data/                       # Project datasets (Ignored in Git except for samples)
│   ├── raw/                    # Raw PDFs & original images downloaded from Vietnamica
│   ├── processed_text/         # 3 separate text folders (Sino-Nom, Transliteration, Translation)
│   └── dataset/                # Final output deliverables
│       ├── images/             # Page images (Max edge <= 1024px)
│       ├── patches/            # Cropped line images (Height <= 96px, rotated 90° CCW)
│       ├── det_gt.txt          # Detection ground truth (Bounding box coordinates)
│       ├── rec_gt.txt          # Recognition ground truth (Patch path - Sino-Nom text)
│       └── results.xlsx        # Consolidated output metadata
│
├── src/                        # Core source code
│   ├── config.py               # Global configurations (Paths, image resize thresholds, etc.)
│   ├── text_processing.py      # Raw text extraction & cleaning module (Member 1)
│   ├── image_ocr.py            # Image processing & PaddleOCR pipeline (Member 2)
│   ├── alignment.py            # BertAlign / Levenshtein alignment algorithm (Member 3)
│   └── utils.py                # Helper functions for I/O operations and formatting
│
├── .gitignore                  # Git ignore rules for large data files
├── requirements.txt            # Python dependencies
├── main.py                     # Entry point executing the full end-to-end pipeline
└── README.md                   # Project documentation

## Coding & Naming Conventions

## Member 1: File src/text_processing.py

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

## Member 2: File src/image_ocr.py

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

## Member 3: File src/alignment.py and src/exporter.py

def align_ocr_with_ground_truth(ocr_boxes: list[dict], han_text_list: list[str]) -> list[dict]:
    """Gióng hàng chữ OCR từ ảnh với bản chữ Hán chuẩn (Dataset 1)."""
    pass

def align_translation_sentences(han_list: list[str], phien_am_list: list[str], dich_list: list[str]) -> list[dict]:
    """Gióng hàng theo cặp câu 3 thành phần (Dataset 2)."""
    pass


# src/exporter.py

def export_det_gt(aligned_data: list, output_file: str = "det_gt.txt") -> None:
    """Xuất file det_gt.txt theo đúng định dạng JSON points."""
    pass

def export_rec_gt(aligned_data: list, output_file: str = "rec_gt.txt") -> None:
    """Xuất file rec_gt.txt (đường dẫn patch \t chữ Hán)."""
    pass

def export_results_excel(aligned_data: list, output_file: str = "results.xlsx") -> None:
    """Xuất file results.xlsx chứa các cột Image ID, Patch ID, Bounding Box, SinoNom OCR."""
    pass

3. Pipeline & Workflow
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

4. Installation & Usage
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

5. Final Deliverables
The final submission folder sent to the Lab/Instructors includes:

 a. 3 Text Folders: Cleaned Sino-Nom, Transliteration, and Translation text files for each stele.

 b. OCR Dataset: Standardized directory structure containing images/, patches/, det_gt.txt, rec_gt.txt, and results.xlsx.

 c. Translation Dataset: Aligned triplets of Sino-Nom – Transliteration – Translation sentences.

 d. Cropped Patches Folder: Preprocessed line crops for training recognition models.

 e. Summary Report: Quantitative metrics on the total number of bounding boxes and aligned sentence pairs.

6. Team Roles & Contribution Breakdown

- Member 1: Lê Trần Khánh Duy - Text Pipeline (29%): Responsible for raw PDF OCR via Google Vision, text filtering, and organizing the 3 text ground-truth datasets.

- Member 2: Huỳnh Đức Dương - Image & Detection Pipeline (36%): Responsible for image acquisition/scaling, running PaddleOCR V6, manual bounding box corrections, stele header re-ordering, and patch extraction.

- Member 3: Lê Quí Thịnh - Alignment, Base Infrastructure & Export (35%): Responsible for repository setup/coding conventions, alignment algorithm integration, automated dataset generation, and final metric reporting.

Data Confidentiality Notice: All materials provided by CLC Lab are restricted to internal academic use only. Redistribution or sharing outside the institutional domain is strictly prohibited.
