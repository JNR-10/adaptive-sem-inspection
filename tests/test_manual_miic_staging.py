from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

from seminspect.utils.io import ArtifactError


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("manual_miic_staging", ROOT / "scripts" / "stage_miic_manual_bundle.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _archive(filename: str, payload: bytes) -> dict[str, object]:
    return {
        "filename": filename,
        "expected_bytes": len(payload),
        "md5": hashlib.md5(payload).hexdigest(),
    }


class ManualMiicStagingTest(unittest.TestCase):
    def test_zip_staging_verifies_and_reuses_declared_files(self) -> None:
        payloads = {"Anomaly_train.rar": b"archive", "readme.txt": b"license"}
        archives = [_archive(name, payload) for name, payload in payloads.items()]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "official-download.zip"
            with ZipFile(bundle, "w") as output:
                for name, payload in payloads.items():
                    output.writestr(f"MIIC/{name}", payload)
            destination = root / "raw"
            with redirect_stdout(io.StringIO()):
                MODULE.stage_bundle(bundle, destination, archives)
            self.assertEqual((destination / "Anomaly_train.rar").read_bytes(), b"archive")
            self.assertEqual((destination / "readme.txt").read_bytes(), b"license")
            with redirect_stdout(io.StringIO()):
                MODULE.stage_bundle(bundle, destination, archives)

    def test_invalid_manual_content_is_not_promoted(self) -> None:
        expected = _archive("Anomaly_train.rar", b"expected")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "official-download.zip"
            with ZipFile(bundle, "w") as output:
                output.writestr("MIIC/Anomaly_train.rar", b"wrong")
            destination = root / "raw"
            with self.assertRaises(ArtifactError):
                MODULE.stage_bundle(bundle, destination, [expected])
            self.assertFalse((destination / "Anomaly_train.rar").exists())

    def test_directory_shards_are_streamed_and_verified(self) -> None:
        payload = b"verified archive payload"
        archive = _archive("Anomaly_train.rar", payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "downloaded-files"
            source.mkdir()
            (source / "Anomaly_train.rar.part-000").write_bytes(payload[:8])
            (source / "Anomaly_train.rar.part-001").write_bytes(payload[8:])
            destination = root / "raw"
            with redirect_stdout(io.StringIO()):
                MODULE.stage_bundle(source, destination, [archive])
            self.assertEqual((destination / "Anomaly_train.rar").read_bytes(), payload)

    def test_directory_shards_require_consecutive_indexes(self) -> None:
        payload = b"verified archive payload"
        archive = _archive("Anomaly_train.rar", payload)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "downloaded-files"
            source.mkdir()
            (source / "Anomaly_train.rar.part-000").write_bytes(payload[:8])
            (source / "Anomaly_train.rar.part-002").write_bytes(payload[8:])
            with self.assertRaises(ArtifactError):
                MODULE.stage_bundle(source, root / "raw", [archive])
