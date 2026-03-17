from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts.authority import apply_reference_io_overlay


class ReferenceIoOverlayTests(unittest.TestCase):
    def test_apply_reference_io_overlay_patches_only_representation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = pathlib.Path(temp_dir)
            control_dict = case_dir / "system" / "controlDict"
            control_dict.parent.mkdir(parents=True, exist_ok=True)
            control_dict.write_text(
                "\n".join(
                    [
                        "application     incompressibleVoF;",
                        "writeFormat     binary;",
                        "writeCompression on;",
                        "writePrecision  6;",
                        "timePrecision   6;",
                        "deltaT          1e-08;",
                        "endTime         2e-08;",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            overlay = apply_reference_io_overlay(case_dir)

            patched_text = control_dict.read_text(encoding="utf-8")

        self.assertIn("writeFormat     ascii;", patched_text)
        self.assertIn("writeCompression off;", patched_text)
        self.assertIn("writePrecision  12;", patched_text)
        self.assertIn("timePrecision   12;", patched_text)
        self.assertIn("deltaT          1e-08;", patched_text)
        self.assertIn("endTime         2e-08;", patched_text)
        self.assertEqual(
            overlay["policy"],
            {
                "write_format": "ascii",
                "write_compression": "off",
                "write_precision": 12,
                "time_precision": 12,
            },
        )
        self.assertEqual(overlay["changes"]["writeFormat"]["before"], "binary")
        self.assertEqual(overlay["changes"]["writeFormat"]["after"], "ascii")
        self.assertEqual(overlay["changes"]["writeCompression"]["before"], "on")
        self.assertEqual(overlay["changes"]["writeCompression"]["after"], "off")
        self.assertTrue(overlay["preserves_numerics"])

    def test_apply_reference_io_overlay_can_write_json_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = pathlib.Path(temp_dir)
            control_dict = case_dir / "system" / "controlDict"
            control_dict.parent.mkdir(parents=True, exist_ok=True)
            control_dict.write_text("application incompressibleVoF;\n", encoding="utf-8")
            json_out = case_dir / "reference_freeze_overlay.json"

            overlay = apply_reference_io_overlay(case_dir, json_out=json_out)
            written_overlay = json.loads(json_out.read_text(encoding="utf-8"))

        self.assertEqual(written_overlay, overlay)
        self.assertEqual(written_overlay["control_dict"], "system/controlDict")
        self.assertEqual(
            written_overlay["changes"]["writeFormat"]["before"],
            None,
        )

    def test_apply_reference_io_overlay_preserves_nested_json_out_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_root = pathlib.Path(temp_dir)
            case_dir = bundle_root / "case"
            control_dict = case_dir / "system" / "controlDict"
            control_dict.parent.mkdir(parents=True, exist_ok=True)
            control_dict.write_text("application incompressibleVoF;\n", encoding="utf-8")
            overlay_artifact = "artifacts/reference/reference_freeze_overlay.json"
            json_out = pathlib.Path("..") / overlay_artifact

            overlay = apply_reference_io_overlay(
                case_dir,
                json_out=json_out,
                overlay_artifact=overlay_artifact,
            )
            written_overlay = json.loads((bundle_root / overlay_artifact).read_text(encoding="utf-8"))

        self.assertEqual(overlay["overlay_artifact"], overlay_artifact)
        self.assertEqual(written_overlay["overlay_artifact"], overlay_artifact)


if __name__ == "__main__":
    unittest.main()
