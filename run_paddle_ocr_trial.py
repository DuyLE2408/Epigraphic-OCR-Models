"""Run this file with .venv-ocr to test PP-OCRv6 separately from the main pipeline."""

import argparse
from importlib.util import find_spec
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    # Windows redirected terminals may default to cp1252, which cannot print
    # Vietnamese messages. Also covers SDK logs without a system-wide change.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Thử PP-OCRv6 local riêng; không gọi API trường.")
    parser.add_argument("--input", nargs="+", type=Path, help="Một hoặc vài ảnh JPG/PNG, hoặc PDF một trang")
    parser.add_argument("--model-size", choices=("medium", "small", "tiny"), default="medium")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="Paddle: cpu hoặc gpu:0")
    parser.add_argument("--rec-score-thresh", type=float, default=0.0)
    parser.add_argument("--det-max-side", type=int, help="Giới hạn xử lý bên trong detector; không sửa kích thước ảnh nguồn")
    parser.add_argument("--det-model-dir", type=Path, help="Model detection đã tải về (tùy chọn)")
    parser.add_argument("--rec-model-dir", type=Path, help="Model recognition đã tải về (tùy chọn)")
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--no-layout", action="store_true", help="So sánh với V6 gốc trên toàn ảnh")
    parser.add_argument("--layout-python", type=Path, default=ROOT / ".venv-layout/Scripts/python.exe")
    parser.add_argument("--layout-model", type=Path, default=ROOT / "models/layout/best.pt")
    parser.add_argument("--layout-device", default="cpu", help="Layout: cpu hoặc cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/image_pipeline/paddle_ocr_trial")
    args = parser.parse_args(argv)
    missing = [name for name in ("paddleocr", "paddle", "PIL", "pymupdf") if find_spec(name) is None]
    if missing:
        print("Python đang dùng:", sys.executable)
        print("Thiếu thư viện:", ", ".join(missing))
        print("Dùng môi trường .venv-ocr; cài requirements-ocr.txt theo docs/paddle_ocr_trial.md.")
        print("Chưa chạy OCR, chưa tải model hoặc thay đổi dữ liệu.")
        return 2
    from src.paddle_ocr_trial import model_settings, run_trial
    try:
        settings = model_settings(args.model_size, args.cpu_threads, args.rec_score_thresh,
                                  args.det_max_side, args.det_model_dir, args.rec_model_dir, device=args.device)
        region_processor = None
        if not args.no_layout:
            from src.local_layout import LocalLayout
            from src.paddle_layout_ocr import LayoutRegionOCR
            region_processor = LayoutRegionOCR(LocalLayout(args.layout_python, args.layout_model,
                                                          device=args.layout_device))
        sources = args.input
        if sources is None:
            if sys.platform != "win32":
                raise ValueError("Colab/Linux: truyền --input với đường dẫn ảnh; không dùng hộp thoại Windows.")
            from tkinter import Tk, filedialog
            window = Tk()
            window.withdraw()
            try:
                selected = filedialog.askopenfilenames(
                    parent=window, title="Thử PaddleOCR V6 — chọn một hoặc vài ảnh/PDF",
                    initialdir=str(ROOT / "data/raw"),
                    filetypes=[("Ảnh hoặc PDF một trang", "*.jpg *.jpeg *.png *.pdf"), ("Tất cả", "*.*")])
            finally:
                window.destroy()
            sources = [Path(p) for p in selected]
        if not sources:
            print("Đã hủy chọn ảnh.")
            return 0
        report = run_trial(sources, args.output_root, settings,
                           render_dpi=args.render_dpi, region_processor=region_processor)
        print("Tổng kết:", report["summary"])
        print("Báo cáo:", Path(report["run_dir"]) / "summary.md")
        if report["status"] == "interrupted":
            return 130
        return 1 if report["summary"]["failed"] or report["summary"]["no_recognized_text"] else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Không chạy được: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
