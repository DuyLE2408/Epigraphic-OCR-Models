"""Verify a portable bundle cannot accidentally collect Lab data or weights."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from scripts.build_colab_paddle_bundle import SOURCE_FILES, build_bundle


class ColabBundleTests(unittest.TestCase):
    def test_only_allowlisted_code_is_packaged_even_with_lab_files_nearby(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in SOURCE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic source only", encoding="utf-8")
            for name in (".env", "models/layout/best.pt", "data/raw/private.jpg",
                         "data/processed_text/private/Han.txt", "src/school_ocr.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("SYNTHETIC_EXCLUDED_CONTENT", encoding="utf-8")
            destination = root / "output.zip"
            with patch("scripts.build_colab_paddle_bundle.ROOT", root):
                build_bundle(destination)
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(set(archive.namelist()), set(SOURCE_FILES) | {"bundle_manifest.json"})
                manifest = json.loads(archive.read("bundle_manifest.json"))
                for name in SOURCE_FILES:
                    content = archive.read(name)
                    self.assertNotIn(b"SYNTHETIC_EXCLUDED_CONTENT", content)
                    self.assertEqual(manifest["files"][name], hashlib.sha256(content).hexdigest())
            before = destination.read_bytes()
            with patch("scripts.build_colab_paddle_bundle.ROOT", root), self.assertRaises(FileExistsError):
                build_bundle(destination)
            self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
