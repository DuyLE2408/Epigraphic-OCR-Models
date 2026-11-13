"""Build an allowlisted source-only ZIP. Never collect data, weights or credentials."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "run_paddle_ocr_trial.py",
    "src/paddle_ocr_trial.py",
    "src/paddle_layout_ocr.py",
    "src/local_layout.py",
    "src/image_ocr.py",
    "src/image_paths.py",
    "src/colab_runtime.py",
    "requirements-colab-layout.txt",
    "requirements-colab-ocr.txt",
    "notebooks/paddle_ocr_v6_colab.ipynb",
    "docs/colab_paddle_trial.md",
)


def build_bundle(destination):
    destination = Path(destination)
    payload = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    manifest = {
        "scope": "PaddleOCR V6 + local Title/Content/Notes; source code only",
        "excluded": ["Lab documents", "images", "model weights", "credentials", "school API pipeline"],
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--output", type=Path, default=ROOT / "data/image_pipeline/colab_bundle" /
                        f"paddle_v6_colab_{stamp}_{uuid4().hex[:8]}.zip")
    args = parser.parse_args()
    print(build_bundle(args.output).resolve())
