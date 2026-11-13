"""CPU-only tests for notebook routing, source scope and output preservation."""

import ast
import base64
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts.build_colab_notebook import notebook
from scripts.build_colab_paddle_bundle import build_bundle
from src import colab_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]


def cell_source(index):
    return "".join(notebook()["cells"][index]["source"])


class ColabRuntimeTests(unittest.TestCase):
    def test_notebook_is_reproducible_executable_python_and_has_no_saved_outputs(self):
        saved = json.loads((ROOT / "notebooks/paddle_ocr_v6_colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(saved, notebook())
        defaults = {}
        exec(cell_source(1), defaults)
        self.assertEqual(defaults["CODE_SOURCE"], "GitHub")
        self.assertEqual(defaults["STORAGE_MODE"], "Upload")
        identifiers = []
        for cell in saved["cells"]:
            identifiers.append(cell["id"])
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertNotIn("drive.mount(", "".join(cell_source(i) for i in (2, 3, 4, 6, 7, 8)))

    def test_github_downloads_only_code_at_one_commit_and_never_persists_token(self):
        secret = "TEST_SECRET_NOT_PERSISTED"
        calls = []
        headers = {}
        sha = "a" * 40
        class Response:
            def __init__(self, value, status=200):
                self.value = value
                self.status_code = status
            def json(self):
                return self.value
        def get(url, **kwargs):
            self.assertFalse(kwargs["allow_redirects"])
            calls.append((url, kwargs))
            if "Authorization" not in headers:
                self.assertIn("/git/ref/heads/", url)
                return Response({}, status=404)
            self.assertEqual(headers["Authorization"], "Bearer " + secret)
            if "/git/ref/heads/" in url:
                return Response({"object": {"sha": sha}})
            name = url.split("/contents/", 1)[1]
            self.assertIn(name, namespace["REQUIRED_FILES"])
            data = (ROOT / name).read_bytes()
            return Response({"type": "file", "encoding": "base64",
                             "content": base64.encodebytes(data).decode(),
                             "sha": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()})
        session = Mock(headers=headers)
        session.get.side_effect = get
        requests = types.ModuleType("requests")
        requests.Session = lambda: session
        colab = types.ModuleType("google.colab")
        colab.userdata = types.SimpleNamespace(get=lambda _: secret)
        with TemporaryDirectory() as temp:
            namespace = {}
            exec(cell_source(1), namespace)
            namespace["WORK_ROOT"] = Path(temp)
            with patch.dict(sys.modules, {"requests": requests, "google.colab": colab}), redirect_stdout(StringIO()) as output:
                exec(cell_source(2), namespace)
            self.assertNotIn(secret, output.getvalue())
            self.assertNotIn("Authorization", headers)
            self.assertEqual(namespace["SOURCE_VERSION"], sha)
            downloaded = sorted(p.relative_to(namespace["CODE_ROOT"]).as_posix()
                                for p in namespace["CODE_ROOT"].rglob("*") if p.is_file())
            self.assertEqual(downloaded, sorted(namespace["REQUIRED_FILES"]))
            for path in namespace["CODE_ROOT"].rglob("*"):
                if path.is_file():
                    self.assertNotIn(secret.encode(), path.read_bytes())
            for url, arguments in calls:
                if "/contents/" in url:
                    self.assertEqual(arguments["params"], {"ref": sha})

    def test_public_github_never_reads_secrets_or_prompts_for_a_token(self):
        headers = {}
        sha = "b" * 40
        def get(url, **kwargs):
            self.assertNotIn("Authorization", headers)
            if "/git/ref/heads/" in url:
                data = {"object": {"sha": sha}}
            else:
                content = b"# synthetic source only\n"
                data = {"type": "file", "encoding": "base64",
                        "content": base64.b64encode(content).decode(),
                        "sha": hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()}
            return types.SimpleNamespace(status_code=200, json=lambda: data)
        session = Mock(headers=headers)
        session.get.side_effect = get
        requests = types.ModuleType("requests")
        requests.Session = lambda: session
        colab = types.ModuleType("google.colab")
        secret_reader = Mock(side_effect=AssertionError("Public code must not request secrets"))
        colab.userdata = types.SimpleNamespace(get=secret_reader)
        with TemporaryDirectory() as temp:
            namespace = {}
            exec(cell_source(1), namespace)
            namespace["WORK_ROOT"] = Path(temp)
            with patch.dict(sys.modules, {"requests": requests, "google.colab": colab}), \
                 patch("getpass.getpass", side_effect=AssertionError("Unexpected token prompt")), \
                 redirect_stdout(StringIO()):
                exec(cell_source(2), namespace)
            self.assertEqual(namespace["SOURCE_VERSION"], sha)
            secret_reader.assert_not_called()

    def test_zip_mode_loads_allowlisted_source_and_upload_mode_needs_no_drive_auth(self):
        colab = types.ModuleType("google.colab")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = build_bundle(root / "code.zip")
            colab.files = types.SimpleNamespace(upload=lambda **_: {str(bundle): b""})
            namespace = {}
            exec(cell_source(1), namespace)
            namespace.update(WORK_ROOT=root / "runtime", CODE_SOURCE="ZIP", STORAGE_MODE="Upload")
            with patch.dict(sys.modules, {"google.colab": colab}), redirect_stdout(StringIO()):
                exec(cell_source(2), namespace)
                exec(cell_source(3), namespace)
            self.assertIsNone(namespace["storage"])
            self.assertTrue((namespace["workspace"] / "results").is_dir())
            self.assertTrue((namespace["CODE_ROOT"] / "src/paddle_layout_ocr.py").is_file())
            self.assertFalse((namespace["CODE_ROOT"] / "src/school_ocr.py").exists())

    def test_gpu_commands_and_environment_setup_never_install_paddle_in_layout_python(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ocr_python = root / "ocr/bin/python"
            ocr_python.parent.mkdir(parents=True)
            ocr_python.touch()
            with patch("src.colab_runtime.version", return_value="2.14.1+cu130"), \
                 patch("src.colab_runtime.stream_command", return_value=0) as stream:
                runtime.setup_environments(root, ocr_python)
            commands = [list(map(str, call.args[0])) for call in stream.call_args_list]
            for command in commands:
                if command[0] == sys.executable:
                    self.assertFalse(any("paddlepaddle" in arg for arg in command))
            engine = next(c for c in commands if "paddlepaddle-gpu==3.2.0" in c)
            self.assertEqual(engine[0], str(ocr_python))
            self.assertIn("https://www.paddlepaddle.org.cn/packages/stable/cu126/", engine)
            command = runtime.pipeline_command(root, ocr_python, sys.executable, root / "best.pt",
                                               [root / "input with spaces.jpg"], root / "results")
            self.assertEqual(command[command.index("--device") + 1], "gpu:0")
            self.assertEqual(command[command.index("--layout-device") + 1], "cuda:0")
            self.assertIn(str(root / "input with spaces.jpg"), command)

    def test_one_rejects_ambiguous_names_all_filters_unrelated_files(self):
        paths = [Path("book-a/00099.jpg"), Path("book-b/00099.jpg"), Path("notes.txt")]
        with self.assertRaises(ValueError):
            runtime.select_inputs(paths, "One")
        self.assertEqual(len(runtime.select_inputs(paths, "All")), 2)
        self.assertEqual(runtime.select_inputs(paths[:1], "One"), paths[:1])
        self.assertEqual(runtime.select_inputs(paths, "One", "book-b/00099.jpg"), paths[1:2])
        with self.assertRaises(ValueError):
            runtime.matches_input_name(paths[0], "../00099.jpg")

    def test_drive_lists_only_requested_folder_with_pagination_and_rejects_unsafe_names(self):
        service = Mock()
        service.files.return_value.list.return_value.execute.side_effect = [
            {"nextPageToken": "page-2", "files": [{"id": "image-id", "name": "00099.jpg", "mimeType": "image/jpeg"}]},
            {"files": [{"id": "text-id", "name": "Han.txt", "mimeType": "text/plain"}]},
        ]
        storage = runtime.DriveStorage(service)
        rows = storage.input_files("project-inputs")
        self.assertEqual([r["name"] for r in rows], ["00099.jpg"])
        for call in service.files.return_value.list.call_args_list:
            self.assertIn("'project-inputs' in parents", call.kwargs["q"])
        with self.assertRaises(ValueError):
            runtime._drive_name("../outside")

    def test_explicit_upload_uses_selected_image_but_directory_lookup_keeps_name_filter(self):
        chosen = Path("inputs/18827.jpg")
        self.assertEqual(runtime.select_inputs([chosen], "One", "00099.jpg", selected_by_upload=True), [chosen])
        with self.assertRaises(ValueError):
            runtime.select_inputs([chosen], "One", "00099.jpg")
        with self.assertRaises(ValueError):
            runtime.select_inputs([chosen, Path("inputs/17512.jpg")], "One", selected_by_upload=True)
        self.assertEqual(runtime.select_inputs([chosen, Path("inputs/17512.jpg")], "All",
                                               selected_by_upload=True),
                         [Path("inputs/17512.jpg"), chosen])

    def test_notebook_accepts_an_uploaded_image_with_a_name_other_than_the_example(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "inputs").mkdir()
            image = root / "inputs/18827.jpg"
            image.write_bytes(b"synthetic selection, no model execution")
            namespace = dict(Path=Path, sources=[], storage=None, LOCAL_INPUT_DIR="",
                             RUN_MODE="One", IMAGE_NAME="00099.jpg", workspace=root,
                             runtime=runtime, files=types.SimpleNamespace(upload=lambda **_: {str(image): b""}))
            with redirect_stdout(StringIO()) as output:
                exec(cell_source(7), namespace)
            self.assertEqual(namespace["sources"], [image])
            self.assertIn("18827.jpg", output.getvalue())

    def test_drive_download_cleans_only_its_own_partial_file_and_keeps_existing_file(self):
        module = types.ModuleType("googleapiclient.http")
        class Downloader:
            def __init__(self, stream, request, **kwargs):
                self.stream = stream
            def next_chunk(self, **kwargs):
                self.stream.write(b"partial")
                raise OSError("synthetic transfer failure")
        module.MediaIoBaseDownload = Downloader
        with TemporaryDirectory() as temp, patch.dict(sys.modules, {"googleapiclient.http": module}):
            path = Path(temp) / "model.pt"
            storage = runtime.DriveStorage(Mock())
            with self.assertRaises(OSError):
                storage.download("id", path)
            self.assertFalse(path.exists())
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                storage.download("id", path)
            self.assertEqual(path.read_bytes(), b"existing")

    def test_failed_pipeline_keeps_zip_even_when_drive_save_also_fails(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = runtime.new_workspace(root)
            fake_runtime = types.SimpleNamespace(
                pipeline_command=Mock(return_value=["not-executed"]),
                stream_command=Mock(return_value=1), archive_results=runtime.archive_results,
            )
            namespace = dict(runtime=fake_runtime, CODE_ROOT=root, OCR_PYTHON=root / "python",
                             sys=sys, model_path=root / "best.pt", sources=[root / "test.jpg"],
                             workspace=workspace, REPOSITORY="owner/repo", BRANCH="Duong",
                             SOURCE_VERSION="synthetic", RUN_MODE="One", json=json,
                             storage=Mock(), drive_outputs="outputs")
            namespace["storage"].upload.side_effect = OSError("synthetic Drive failure")
            with redirect_stdout(StringIO()) as output:
                exec(cell_source(8), namespace)
            self.assertIn("ZIP vẫn còn tại", output.getvalue())
            with zipfile.ZipFile(namespace["result_zip"]) as archive:
                record = json.loads(archive.read("colab_run.json"))
                self.assertEqual(record["exit_code"], 1)

    def test_result_archive_does_not_collect_nearby_models_or_inputs_and_is_exclusive(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "results").mkdir()
            (root / "results/report.json").write_text("{}")
            (root / "best.pt").write_text("excluded synthetic model")
            (root / ".env").write_text("excluded synthetic secret")
            result = runtime.archive_results(root / "results", root / "result.zip")
            with zipfile.ZipFile(result) as archive:
                self.assertEqual(archive.namelist(), ["report.json"])
            with self.assertRaises(FileExistsError):
                runtime.archive_results(root / "results", result)
            with self.assertRaises(ValueError):
                runtime.archive_results(root / "results", root / "results/recursive.zip")


if __name__ == "__main__":
    unittest.main()
