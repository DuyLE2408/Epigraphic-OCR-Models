"""PDF/JPEG image normalization for Member 2.

This stage uses only Pillow and PyMuPDF. ``resize_page_image`` accepts the
README arguments but currently returns a Pillow RGB image, not a NumPy array.
The ndarray/BGR adapter and OCR/box/patch functions belong to later stages.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from html import escape
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote

from PIL import Image, ImageOps
import pymupdf

from src.image_paths import normalization_paths, prepare_folder


APP_ROOT = Path(__file__).resolve().parents[1]


def normalize_image(image: Image.Image, max_dim: int | None = 1024) -> Image.Image:
    """Return RGB; None preserves dimensions, an explicit limit shrinks for export."""
    if max_dim is not None and (isinstance(max_dim, bool) or not isinstance(max_dim, int) or max_dim < 1):
        raise ValueError("max_dim must be a positive integer or None")
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", image.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
    else:
        rgb = image.convert("RGB")
    scale = 1.0 if max_dim is None else min(1.0, max_dim / max(rgb.size))
    size = tuple(max(1, round(side * scale)) for side in rgb.size)
    if size != rgb.size:
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    return rgb


def resize_page_image(image_path: str, max_dim: int = 1024) -> Image.Image:
    """Read a raster image and normalize it; return Pillow RGB for this stage.

    Positional/keyword arguments match the README. NumPy/BGR return compatibility
    is deferred to the OCR stage to avoid adding NumPy to this experiment.
    Raster EXIF orientation is respected; PDF orientation is handled separately.
    """
    with Image.open(image_path) as image:
        return normalize_image(ImageOps.exif_transpose(image), max_dim)


def sort_tran_bia_boxes(boxes: list) -> list:
    """README adapter: input is confirmed header boxes; sort rows right-to-left."""
    from src.box_postprocess import sort_boxes
    return sort_boxes([dict(box, region=box.get('region', 'header')) for box in boxes])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_page(page: pymupdf.Page) -> dict:
    images = []
    for info in page.get_image_info(xrefs=True):
        images.append({key: info[key] for key in (
            "xref", "width", "height", "bbox", "transform", "cs-name", "bpc"
        )})
        images[-1]["has_mask"] = bool(info.get("has-mask", False))
    return {
        "page_rect_points": list(page.rect),
        "media_box_points": list(page.mediabox),
        "crop_box_points": list(page.cropbox),
        "page_rotation_degrees": page.rotation,
        "embedded_images": images,
        "paint_events": dict(Counter(entry[0] for entry in page.get_bboxlog())),
        "annotation_count": len(list(page.annots() or [])),
        "widget_count": len(list(page.widgets() or [])),
    }


def _extract_if_full_page(document: pymupdf.Document, page: pymupdf.Page,
                          inspection: dict) -> tuple[Image.Image | None, str]:
    """Extract only a single, unmasked, unrotated, full-page image.

    Any ambiguous layout, margin, clipping, skew, flip, or other painted content
    falls back to rendering the PDF page. Never assume the first image is right.
    """
    images = inspection["embedded_images"]
    if len(images) != 1 or inspection["paint_events"] != {"fill-image": 1}:
        return None, "Page does not paint exactly one image and nothing else"
    if page.rotation or inspection["annotation_count"] or inspection["widget_count"]:
        return None, "Page rotation, annotations, or widgets require rendering"
    info = images[0]
    if info["xref"] <= 0 or info["has_mask"]:
        return None, "Inline or masked image requires rendering"
    if not all(math.isclose(a, b, abs_tol=0.01, rel_tol=0)
               for a, b in zip(info["bbox"], page.rect)):
        return None, "Image does not cover the complete visible page"
    a, b, c, d, _, _ = info["transform"]
    if a <= 0 or d <= 0 or abs(b) > 0.001 or abs(c) > 0.001:
        return None, "Image is rotated, flipped, or skewed in the PDF"
    if not math.isclose(a / info["width"], d / info["height"], rel_tol=1e-4):
        return None, "PDF uses nonuniform image scaling"
    # PDF Decode/Mask entries can alter the embedded image's appearance.
    if any(document.xref_get_key(info["xref"], key)[0] != "null"
           for key in ("Decode", "Mask", "SMask")):
        return None, "PDF image has a Decode or mask entry"
    if info["cs-name"] not in ("DeviceGray", "DeviceRGB"):
        return None, "PDF image colorspace requires page rendering"
    # Reject clipping paths, even when the image's reported bbox fills the page.
    if any(item["type"] == "clip" for item in page.get_drawings(extended=True)):
        return None, "Page contains a clipping path"
    extracted = document.extract_image(info["xref"])
    if not extracted or extracted.get("smask", 0):
        return None, "Embedded image could not be extracted without a mask"
    with Image.open(BytesIO(extracted["image"])) as image:
        if image.size != (info["width"], info["height"]):
            return None, "Extracted image dimensions differ from PDF metadata"
        if image.getexif().get(274, 1) != 1:
            return None, "Embedded EXIF orientation differs from PDF placement"
        return image.copy(), "One unmasked image covers the full page at uniform scale"


def load_single_page_pdf(pdf_path: Path, render_dpi: int = 300) -> tuple[Image.Image, dict]:
    """Load one valid PDF page locally, preferring exact embedded extraction."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input must be a PDF file")
    if isinstance(render_dpi, bool) or not isinstance(render_dpi, int) or render_dpi < 72:
        raise ValueError("render_dpi must be an integer of at least 72")
    with pymupdf.open(pdf_path) as document:
        if not document.is_pdf or document.needs_pass or document.is_repaired:
            raise ValueError("PDF is invalid, encrypted, or required repair; stopped")
        if document.page_count != 1:
            raise ValueError(f"Expected exactly one PDF page; found {document.page_count}")
        page = document[0]
        inspection = _inspect_page(page)
        image, reason = _extract_if_full_page(document, page, inspection)
        method = "extract"
        if image is None:
            pixmap = page.get_pixmap(dpi=render_dpi, colorspace=pymupdf.csRGB,
                                     alpha=False, annots=True)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            method = "render"
        inspection.update(page_count=1, extraction_method=method, extraction_reason=reason)
        return image, inspection


def load_source_image(input_path: Path, render_dpi: int = 300) -> tuple[Image.Image, dict]:
    """Read one actual PDF or JPEG; respect JPEG EXIF without guessing rotation."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        image, inspection = load_single_page_pdf(input_path, render_dpi)
        inspection.update(source_type="pdf", source_page=1,
                          load_method=inspection["extraction_method"])
        return image, inspection
    if suffix not in {".jpg", ".jpeg"}:
        raise ValueError("Supported input types are PDF, JPG, and JPEG")
    with Image.open(input_path) as image:
        if image.format != "JPEG":
            raise ValueError("File extension is JPG/JPEG, but its content is not JPEG")
        image.load()
        encoded_size = list(image.size)
        orientation = image.getexif().get(274, 1)
        oriented = ImageOps.exif_transpose(image)
        return oriented.copy(), {
            "source_type": "jpeg", "source_page": None,
            "load_method": "direct_image", "encoded_image_size": encoded_size,
            "exif_orientation": orientation,
            "exif_orientation_applied": orientation in {2, 3, 4, 5, 6, 7, 8},
        }


def _write_new_files(outputs: dict[Path, bytes]) -> None:
    """Write exclusively; never overwrite. Remove only our own partial outputs."""
    for path in outputs:
        if os.path.lexists(path):
            raise FileExistsError(f"Output already exists; stopped: {path}")
    created = []
    try:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                created.append(path)
                stream.write(content)
    except Exception:
        for path in reversed(created):
            path.unlink()
        raise


def _relative_path(path: Path, app_root: Path) -> str:
    return Path(os.path.relpath(path, app_root)).as_posix()


def _format_image_number(image_number: int | str) -> str:
    """An image index is independent of the optional document page."""
    if (isinstance(image_number, bool) or not isinstance(image_number, (int, str))
            or not re.fullmatch(r"[0-9]{1,3}", str(image_number))
            or not 1 <= int(image_number) <= 999):
        raise ValueError("image_number must be an image index from 1 to 999")
    return f"{int(image_number):03d}"


def _provided_page_number(value: int | str | None) -> int | None:
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, str))
            or not re.fullmatch(r"[0-9]+", str(value)) or int(value) < 1):
        raise ValueError("page_number must be a positive supplied reference page or None")
    return int(value)


def _select_image_number(input_file: Path, book_id: str, stele_id: str,
                         app_root: Path) -> str:
    """Reuse source metadata or allocate the next image index within one stele."""
    private_dir = prepare_folder(app_root, book_id, stele_id)
    used, mapped = set(), set()
    pattern = re.compile(rf'{re.escape(stele_id)}\.([0-9]{{3}})\.json')
    for path in normalization_paths(app_root, book_id, stele_id):
        match = pattern.fullmatch(path.name)
        record = json.loads(path.read_text(encoding='utf-8'))
        if match:
            used.add(int(match.group(1)))
        source = record.get('source_path') or record.get('input_pdf')
        original = (app_root / source).resolve() if source else None
        raw_book = (app_root / 'data/raw' / book_id).resolve()
        same_source = original == input_file
        relocated = (original is not None and original.is_relative_to(raw_book)
                     and original.name == input_file.name and input_file.parent == raw_book)
        if same_source or relocated:
            if not match:
                raise FileExistsError(f'Source already has a legacy output; kept unchanged: {path}')
            mapped.add(match.group(1))
    if len(mapped) > 1:
        raise ValueError('Source has multiple image indices; check its metadata')
    if mapped:
        return mapped.pop()
    for folder, expression in (
        (private_dir, '*.jpg'),
        (private_dir, '*.preview.html'),
        (app_root / 'data/image_pipeline' / book_id / stele_id, '*.preview.html'),
    ):
        for path in folder.glob(expression):
            match = re.fullmatch(rf'{re.escape(stele_id)}\.([0-9]{{3}})\.(?:jpg|preview\.html)',
                                 path.name)
            if match:
                used.add(int(match.group(1)))
    return _format_image_number(max(used, default=0) + 1)


def _preview_html(source: Path, jpg: Path, preview: Path, stele_id: str,
                  *, source_type: str = "pdf") -> bytes:
    source_url = quote(_relative_path(source, preview.parent), safe="/")
    jpg_url = quote(_relative_path(jpg, preview.parent), safe="/")
    source_label = "PDF gốc" if source_type == "pdf" else "JPG gốc"
    source_view = (f'<iframe title="PDF gốc" src="{source_url}#page=1"></iframe>'
                   if source_type == "pdf" else
                   f'<p><img alt="JPG gốc" src="{source_url}"></p>')
    html = f"""<!doctype html>
<html lang="vi"><meta charset="utf-8"><title>{escape(stele_id)} — kiểm tra ảnh</title>
<style>body{{font:16px system-ui;margin:24px;background:#eee}}main{{display:flex;
gap:20px;flex-wrap:wrap}}section{{flex:1;min-width:300px}}iframe{{width:100%;height:85vh}}
img{{max-width:100%;height:auto}}a{{color:#164a85}}</style>
<h1>{escape(stele_id)} — {source_label} và JPG chuẩn hóa</h1>
<p>So sánh hướng ảnh, bốn mép và toàn bộ nội dung. Trang này chỉ dùng file cục bộ.</p>
<main><section><h2>{source_label}</h2><a href="{source_url}">Mở nguồn gốc</a>
{source_view}</section>
<section><h2>JPG chuẩn hóa</h2><a href="{jpg_url}">Mở JPG</a>
<p><img alt="Ảnh chuẩn hóa" src="{jpg_url}"></p></section></main></html>"""
    return html.encode("utf-8")


def normalize_single_input(input_path: str | Path, stele_id: str,
                           output_path: str | Path | None = None, *,
                           book_id: str, page_number: int | str | None = None,
                           provided_page: dict | None = None,
                           image_number: int | str | None = None,
                           app_root: Path = APP_ROOT, max_dim: int | None = None,
                           render_dpi: int = 300, preview: bool = False) -> dict:
    """Prepare native-resolution RGB for OCR; supplied pages are metadata only.

    An explicit max_dim remains available for local legacy/export callers.
    The school OCR workflow only accepts the default native-resolution policy.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", book_id):
        raise ValueError("book_id must contain only letters, digits, underscores or hyphens")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", stele_id):
        raise ValueError("stele_id must contain only letters, digits, underscores or hyphens")
    if max_dim is not None and (isinstance(max_dim, bool) or not isinstance(max_dim, int) or max_dim < 1):
        raise ValueError("max_dim must be a positive integer or None")
    app_root = Path(app_root).resolve()
    def resolve(path: str | Path) -> Path:
        path = Path(path)
        return (app_root / path).resolve() if not path.is_absolute() else path.resolve()
    input_file = resolve(input_path)
    if provided_page:
        if (provided_page.get('stele_id') != stele_id
                or resolve(provided_page.get('source_path', '')) != input_file):
            raise ValueError('Supplied page metadata does not belong to the selected source/stele')
        if (provided_page.get('page_number_source') not in {None, 'provided_excel', 'provided_explicit'}
                or (provided_page.get('book_page_number') is not None
                    and provided_page.get('page_number_source') is None)):
            raise ValueError('Only supplied page metadata is accepted; inferred pages are not used')
    reference_page = _provided_page_number(page_number if page_number is not None else
                                           provided_page.get('book_page_number') if provided_page else None)
    number = (_format_image_number(image_number) if image_number is not None else
              _select_image_number(input_file, book_id, stele_id, app_root))
    image_id = f"{stele_id}.{number}"
    filename = f"{image_id}.jpg"
    book_images = prepare_folder(app_root, book_id, stele_id)
    jpg = resolve(output_path or book_images / filename)
    if jpg.name != filename:
        raise ValueError(f"Output filename must be {filename}")
    if jpg.parent != book_images.resolve():
        raise ValueError(f"Output must be inside {book_images.relative_to(app_root).as_posix()}/")
    private_dir = book_images
    metadata_path = private_dir / f"{image_id}.json"
    preview_path = private_dir / f"{image_id}.preview.html"
    targets = [jpg, metadata_path] + ([preview_path] if preview else [])
    for path in targets:
        if os.path.lexists(path):
            raise FileExistsError(f"Output already exists; stopped: {path}")
    legacy_jpg = app_root / "data/dataset" / book_id / "images" / filename
    if max_dim is not None and os.path.lexists(legacy_jpg):
        raise FileExistsError(f"Legacy output already exists; use the pipeline to validate and reuse it: {legacy_jpg}")
    if len({path.resolve() for path in targets}) != len(targets) or input_file in targets:
        raise ValueError("Input and output paths must be distinct")
    original_hash = _sha256(input_file)
    source, inspection = load_source_image(input_file, render_dpi)
    before = source.size
    normalized = normalize_image(source, max_dim)
    copy_jpeg = (max_dim is None and inspection['source_type'] == 'jpeg'
                 and source.mode == 'RGB' and not inspection['exif_orientation_applied'])
    if copy_jpeg:
        # Avoid another lossy JPEG encoding when the source is already suitable.
        jpg_bytes = input_file.read_bytes()
    else:
        encoded = BytesIO()
        normalized.save(encoded, format="JPEG", quality=95, subsampling=0, optimize=True)
        jpg_bytes = encoded.getvalue()
    with Image.open(BytesIO(jpg_bytes)) as check:
        check.load()
        if check.format != "JPEG" or check.mode != "RGB" or check.size != normalized.size:
            raise ValueError("Encoded JPEG failed validation")
    unchanged = _sha256(input_file) == original_hash
    if not unchanged:
        raise RuntimeError("Input changed during processing; no output written")
    metadata = {
        "schema_version": "2.0", "book_id": book_id, "image_id": image_id,
        "stele_id": stele_id, "image_number": number,
        "page_number": reference_page, "book_page_number": reference_page,
        "page_number_source": ('provided_explicit' if page_number is not None else
                               provided_page.get('page_number_source') if provided_page else None),
        "page_scope": ('user_provided' if page_number is not None else
                       provided_page.get('page_scope') if provided_page else None),
        "filename_status": "stele_image_index",
        "book_reference": provided_page,
        "source_path": _relative_path(input_file, app_root),
        "source_sha256": original_hash,
        "output_jpg": _relative_path(jpg, app_root),
        "output_jpg_sha256": hashlib.sha256(jpg_bytes).hexdigest(),
        **inspection,
        "image_mode_before": source.mode, "image_mode_after": "RGB",
        "image_size_before_resize": list(before),
        "image_size_after_resize": list(normalized.size),
        "source_image_size": list(before),
        "image_size": list(normalized.size),
        "coordinate_space": "prepared_image_pixels",
        "resize_scale": {"x": normalized.width / before[0],
                         "y": normalized.height / before[1]},
        "processing": {"max_dim": max_dim,
                       "resolution_policy": "native" if max_dim is None else "limited",
                       "resampling": "LANCZOS" if normalized.size != before else None,
                       "upscale": False, "crop": False, "auto_rotate": False,
                       "contrast_adjustment": False, "sharpen": False,
                       "binarize": False, "encoding": "copy_jpeg" if copy_jpeg else "jpeg_rgb",
                       "jpeg_quality": None if copy_jpeg else 95,
                       "jpeg_subsampling": None if copy_jpeg else 0,
                       "respect_exif_orientation": inspection["source_type"] == "jpeg",
                       "render_dpi": render_dpi if inspection["load_method"] == "render" else None},
        "software": {"pymupdf": pymupdf.VersionBind, "pillow": Image.__version__},
        "source_unchanged": unchanged,
        "metadata_path": _relative_path(metadata_path, app_root),
    }
    if inspection["source_type"] == "pdf":
        # Preserve PDF-only aliases for existing consumers. JPEG has none of these.
        metadata.update(input_pdf=metadata["source_path"], input_pdf_sha256=original_hash,
                        source_pdf_page=1, input_pdf_unchanged=unchanged)
    outputs = {jpg: jpg_bytes}
    if preview:
        metadata["preview_path"] = _relative_path(preview_path, app_root)
        outputs[preview_path] = _preview_html(input_file, jpg, preview_path, stele_id,
                                              source_type=inspection["source_type"])
    outputs[metadata_path] = (json.dumps(metadata, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n").encode("utf-8")
    _write_new_files(outputs)
    return metadata


def archive_previous_records(app_root, book_id, stele_id, image_id):
    """Move only the old per-image JSON/HTML into the preparation stage's history."""
    app_root = Path(app_root).resolve()
    stage = prepare_folder(app_root, book_id, stele_id).resolve()
    old_folder = stage.parent
    archive = stage / "_legacy"
    if not re.fullmatch(rf"{re.escape(stele_id)}\.[0-9]{{3}}", image_id):
        raise ValueError("Cannot archive records for an invalid image ID")
    for suffix in (".json", ".preview.html"):
        source = (old_folder / f"{image_id}{suffix}").resolve()
        target = (archive / source.name).resolve()
        if (source.parent != old_folder or not source.is_relative_to(app_root)
                or target.parent != archive or not target.is_relative_to(app_root)):
            raise ValueError("Archive paths escaped the processing directory")
        if not source.is_file():
            continue
        archive.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _sha256(source) != _sha256(target):
                raise FileExistsError(f"Archive differs; old records kept: {target}")
            source.unlink()
        else:
            source.rename(target)
    return _relative_path(archive / f"{image_id}.json", app_root)


def reuse_normalized_output(metadata, app_root, *, preview=True):
    """Copy a verified old output into its processing stage; keep originals."""
    app_root = Path(app_root).resolve()
    image = (app_root / metadata["output_jpg"]).resolve()
    destination = prepare_folder(app_root, metadata["book_id"], metadata["stele_id"]).resolve()
    if image.parent == destination:
        archive_previous_records(app_root, metadata["book_id"], metadata["stele_id"], metadata["image_id"])
        return metadata
    source = (app_root / (metadata.get("source_path") or metadata["input_pdf"])).resolve()
    raw_book = (app_root / "data/raw" / metadata["book_id"]).resolve()
    if source.is_relative_to(raw_book) and not source.is_file():
        source = raw_book / source.name
    source_hash = metadata.get("source_sha256") or metadata.get("input_pdf_sha256")
    if _sha256(image) != metadata["output_jpg_sha256"] or _sha256(source) != source_hash:
        raise ValueError("Cannot relocate changed source/output")
    match = re.fullmatch(rf"{re.escape(metadata['stele_id'])}\.([0-9]{{3}})\.jpg", image.name)
    if not match or int(match.group(1)) < 1:
        raise ValueError("Legacy filename has no verified stele/image index; kept for review")
    image_id = image.stem
    if metadata.get("image_id", image_id) != image_id:
        raise ValueError("Legacy image ID does not match its filename")
    jpg = destination / f"{image_id}.jpg"
    record = destination / f"{image_id}.json"
    html = destination / f"{image_id}.preview.html"
    source_type = metadata.get("source_type", "pdf" if source.suffix.lower() == ".pdf" else "jpeg")
    updated = dict(metadata, schema_version="1.7", image_id=image_id, image_number=match.group(1),
                   source_path=_relative_path(source, app_root), source_sha256=source_hash,
                   source_type=source_type, source_page=1 if source_type == "pdf" else None,
                   source_unchanged=True, filename_status="stele_image_index",
                   load_method=metadata.get("load_method") or metadata.get("extraction_method"),
                   output_jpg=_relative_path(jpg, app_root),
                   legacy_metadata_path=_relative_path(destination / "_legacy" / f"{image_id}.json", app_root),
                   metadata_path=_relative_path(record, app_root))
    updated.pop("dataset_page_number", None)
    if updated.get("page_number_source") not in {"provided_excel", "provided_explicit"}:
        updated.update(page_number=None, book_page_number=None, page_number_source=None, page_scope=None)
    outputs = {jpg: image.read_bytes()}
    if preview:
        updated["preview_path"] = _relative_path(html, app_root)
        outputs[html] = _preview_html(source, jpg, html, metadata["stele_id"],
                                     source_type=source_type)
    else:
        updated.pop("preview_path", None)
    outputs[record] = (json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_new_files(outputs)
    archive_previous_records(app_root, metadata["book_id"], metadata["stele_id"], image_id)
    return updated
