"""Persistent OCR work queue; completed images stay outside the processing loop."""

from datetime import datetime, timezone
import json
from pathlib import Path

from src.image_batch import _atomic_text, _validated_existing
from src.image_paths import normalization_paths, ocr_folder
from src.page_reference import SUPPORTED_BOOKS, collect_sources, _file_hash


def _signature(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class PipelineProgress:
    def __init__(self, root, parameters, *, max_dim=None, render_dpi=300, recheck=False,
                 target_stage='02_ocr', include_header=False):
        self.root = Path(root).resolve()
        self.path = self.root / "data/image_pipeline/school_api/progress.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.parameters = dict(parameters)
        self.settings = {"max_dim": max_dim, "render_dpi": render_dpi,
                         "resolution_policy": "native" if max_dim is None else "limited",
                         "ocr_parameters": self.parameters}
        self.recheck = recheck
        self.include_header = include_header
        self._needs_box_stage = set()
        if target_stage not in {'02_ocr', '03_filter_sort'}:
            raise ValueError('Unknown target stage')
        self.target_stage = target_stage
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                if (self.data["schema_version"] != "1.0"
                        or not isinstance(self.data["completed"], dict)
                        or not isinstance(self.data["pending"], list)
                        or any(not isinstance(item, dict) or not isinstance(item.get("source_path"), str)
                               for item in self.data["pending"])):
                    raise ValueError("Invalid progress schema")
            except (ValueError, KeyError, TypeError):
                raise ValueError("File progress.json bị lỗi; giữ nguyên để kiểm tra, chưa gọi API.") from None
            if self.data.get("settings") != self.settings:
                raise ValueError("Cấu hình khác progress.json; kiểm tra kết quả trước khi thay cấu hình.")
        else:
            self.data = {"schema_version": "1.0", "target_stage": "02_ocr", "settings": self.settings,
                         "completed": {}, "pending": [], "next_source": None,
                         "last_run_id": None, "last_status": "initialized"}
            self._bootstrap()
            self.save()
        self.data['target_stage'] = target_stage

    def _bootstrap(self):
        """One-time import of verified results made before the queue existed."""
        for book in sorted(SUPPORTED_BOOKS):
            raw = self.root / "data/raw" / book
            if not raw.is_dir():
                continue
            sources = collect_sources(self.root, book)
            for source in sources:
                self.data["pending"].append({**source, "book_id": book, "status": "not_run"})
            for path in normalization_paths(self.root, book):
                try:
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    image_id, stele_id = metadata["image_id"], metadata["stele_id"]
                    folder = ocr_folder(self.root, book, stele_id, image_id)
                    marker = folder / "complete.json"
                    if not marker.is_file():
                        continue
                    saved = json.loads(marker.read_text(encoding="utf-8"))
                    expected = {"image_id": image_id, "image_sha256": metadata["output_jpg_sha256"],
                                "parameters": self.parameters}
                    if saved.get("binding") != expected:
                        continue
                    for key, filename in (("raw", "ocr_raw.json"), ("boxes", "ocr.json"),
                                          ("preview", "bbox_preview.png")):
                        if _file_hash(folder / filename) != saved["sha256"].get(key):
                            raise ValueError("Invalid previous OCR")
                    source = (self.root / metadata["source_path"]).resolve()
                    prior = _validated_existing(source, [(path, metadata)], self.root, book,
                                                self.settings["max_dim"], self.settings["render_dpi"])
                    if prior is None:
                        continue
                    self.data["completed"][source.relative_to(self.root).as_posix()] = {
                        "book_id": book, "stele_id": stele_id, "image_id": image_id,
                        "source_signature": _signature(source),
                        "metadata_path": path.relative_to(self.root).as_posix(),
                        "complete_path": marker.relative_to(self.root).as_posix(),
                    }
                except (OSError, ValueError, TypeError, KeyError):
                    continue  # Pending entries will receive normal validation and a detailed report.
        completed = self.data["completed"]
        self.data["pending"] = [item for item in self.data["pending"]
                                if item["source_path"] not in completed]

    def should_skip(self, source_path, source):
        if self.recheck:
            return False
        saved = self.data["completed"].get(source_path)
        if not isinstance(saved, dict):
            return False
        if saved.get("source_signature") != _signature(source):
            self.data["completed"].pop(source_path)
            return False
        # Missing preparation/OCR stages must also return a source to the queue.
        metadata_path = self.root / saved.get('metadata_path', '')
        if (not metadata_path.is_file() or not metadata_path.with_suffix('.jpg').is_file()
                or not (self.root / saved.get('complete_path', '')).is_file()):
            self.data['completed'].pop(source_path)
            return False
        if self.include_header:
            from src.header_ocr import header_result_is_current
            if not header_result_is_current(self.root, saved.get('book_id'), saved.get('stele_id'),
                                            saved.get('image_id')):
                self.data['completed'].pop(source_path)
                self._needs_box_stage.add(source_path)
                return False
        if self.target_stage == '03_filter_sort':
            from src.box_postprocess import filter_result_is_current
            if not filter_result_is_current(self.root, saved.get('book_id'), saved.get('stele_id'),
                                             saved.get('image_id'),
                                             expected_source='ocr_combined.json' if self.include_header else None):
                self.data['completed'].pop(source_path)
                self._needs_box_stage.add(source_path)
                return False
        return True

    def prepare_queue(self, rows, *, full_scan, run_id):
        old = {item["source_path"]: item for item in self.data["pending"]}
        rank = {key: index for index, key in enumerate(old)}
        rows.sort(key=lambda row: (0 if row['source_path'] in self._needs_box_stage else 1,
                                   rank.get(row["source_path"], len(rank))))
        current = {row["source_path"]: {
            **old.get(row["source_path"], {}), "source_path": row["source_path"],
            "book_id": row["book_id"], "stele_id": row["stele_id"],
            "status": old.get(row["source_path"], {}).get("status", "not_run"),
        } for row in rows}
        self.data["pending"] = (list(current.values()) if full_scan else
                                [item for key, item in old.items() if key not in current] + list(current.values()))
        self.data["last_run_id"] = run_id
        self.data["last_status"] = "running"
        self.save()

    def record(self, row):
        key = row["source_path"]
        box_done = self.target_stage != '03_filter_sort' or row.get('box_status') in {'success', 'skipped_existing'}
        if box_done and row.get("ocr_status") in {"success", "skipped_existing"} and row["status"] in {"success", "skipped_existing"}:
            metadata = json.loads((self.root / row["metadata_path"]).read_text(encoding="utf-8"))
            source = self.root / key
            if _file_hash(source) != metadata["source_sha256"]:
                raise ValueError("Ảnh nguồn đổi trong khi OCR; chưa ghi hoàn thành vào progress.json.")
            self.data["completed"][key] = {
                "book_id": row["book_id"], "stele_id": row["stele_id"],
                "image_id": row["image_id"], "source_signature": _signature(source),
                "metadata_path": row["metadata_path"],
                "complete_path": (ocr_folder(self.root, row["book_id"], row["stele_id"],
                                             row["image_id"]) / "complete.json").relative_to(self.root).as_posix(),
                "box_count": row.get("box_count"),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                'target_stage': self.target_stage,
            }
            self.data["pending"] = [item for item in self.data["pending"] if item["source_path"] != key]
        else:
            self.data["completed"].pop(key, None)
            entry = {k: row[k] for k in ("source_path", "book_id", "stele_id", "status",
                                         "reason", "ocr_status") if k in row}
            for index, item in enumerate(self.data["pending"]):
                if item["source_path"] == key:
                    self.data["pending"][index] = entry
                    break
            else:
                self.data["pending"].append(entry)
        self.save()

    def finish(self, report):
        self.data.update(last_status=report["status"], last_run_id=report["run_id"],
                         resume_at_utc=report.get("resume_at_utc"))
        self.save()

    def save(self):
        self.data['target_stage'] = self.target_stage
        self.data["next_source"] = (self.data["pending"][0]["source_path"]
                                    if self.data["pending"] else None)
        self.data["summary"] = {"completed": len(self.data["completed"]),
                                "pending": len(self.data["pending"])}
        self.data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_text(self.path, json.dumps(self.data, ensure_ascii=False, indent=2) + "\n")
