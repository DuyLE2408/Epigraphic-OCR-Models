"""Per-book, per-stele processing stage paths."""

from pathlib import Path
import re


def stele_folder(root, book_id, stele_id):
    for component in (book_id, stele_id):
        if not isinstance(component, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", component):
            raise ValueError("Invalid book or stele ID")
    return Path(root) / "data/image_pipeline" / book_id / stele_id


def prepare_folder(root, book_id, stele_id):
    return stele_folder(root, book_id, stele_id) / "01_prepare"


def ocr_folder(root, book_id, stele_id, image_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", image_id):
        raise ValueError("Invalid image ID")
    return stele_folder(root, book_id, stele_id) / "02_ocr" / image_id


def filter_sort_folder(root, book_id, stele_id, image_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", image_id):
        raise ValueError("Invalid image ID")
    return stele_folder(root, book_id, stele_id) / "03_filter_sort" / image_id


def combined_ocr_paths(folder):
    return {key: Path(folder) / name for key, name in {
        'raw': 'combined_sources.json', 'boxes': 'ocr_combined.json',
        'preview': 'bbox_combined_preview.png', 'selection': 'header_selection.json',
        'complete': 'complete_combined.json',
    }.items()}


def preferred_ocr_path(folder):
    combined = combined_ocr_paths(folder)
    return (combined['boxes'] if combined['boxes'].exists() or combined['complete'].exists()
            else Path(folder) / 'ocr.json')


def normalization_paths(root, book_id, stele_id=None):
    """New stage records first; retain read access to old normalization records."""
    book = Path(root) / "data/image_pipeline" / book_id
    folders = [book / stele_id] if stele_id is not None else sorted(p for p in book.glob("*") if p.is_dir())
    for folder in folders:
        yield from sorted((folder / "01_prepare").glob("*.json"))
        yield from sorted(folder.glob("*.json"))


def same_normalization(first, second):
    """Copies of one record have the same provenance despite relocated outputs."""
    def identity(record):
        return (record.get("image_id") or Path(record.get("output_jpg", "")).stem,
                record.get("source_sha256") or record.get("input_pdf_sha256"),
                record.get("output_jpg_sha256"))
    values = identity(first)
    return all(value is not None for value in values) and values == identity(second)
