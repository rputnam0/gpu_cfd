from __future__ import annotations

import pathlib
import subprocess
import unittest
from contextlib import redirect_stdout
from io import StringIO

from scripts.authority import (
    load_authority_bundle,
    render_source_audit_note,
    resolve_source_audit_surfaces,
    validate_source_audit_note,
)
from scripts.authority import source_audit


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


class SourceAuditHelperTests(unittest.TestCase):
    def test_resolve_source_audit_surfaces_maps_each_surface_once(self) -> None:
        bundle = load_authority_bundle(repo_root())

        resolved = resolve_source_audit_surfaces(
            bundle,
            ["Alpha transport", "Pressure corrector"],
        )

        self.assertEqual(
            [entry.contract_surface for entry in resolved],
            ["Alpha transport", "Pressure corrector"],
        )
        self.assertIn("alphaPredictor.C", resolved[0].ownership_scope)
        self.assertIn("DeviceMULES.*", resolved[0].ownership_scope)
        self.assertIn("pressureCorrector.C", resolved[1].ownership_scope)

    def test_resolve_source_audit_surfaces_accepts_phase_doc_aliases(self) -> None:
        bundle = load_authority_bundle(repo_root())

        resolved = resolve_source_audit_surfaces(
            bundle,
            ["alphaPredictor", "pressureCorrector", "interfaceProperties", "momentum stage"],
        )

        self.assertEqual(
            [entry.contract_surface for entry in resolved],
            [
                "Alpha transport",
                "Pressure corrector",
                "Interface properties",
                "Momentum predictor",
            ],
        )

    def test_render_source_audit_note_includes_reviewable_fields(self) -> None:
        bundle = load_authority_bundle(repo_root())

        note = render_source_audit_note(
            bundle,
            touched_surfaces=["Alpha transport", "Pressure bridge"],
            review_status="reviewed",
            note_title="Phase 5 Source Audit",
        )

        self.assertIn("# Phase 5 Source Audit", note)
        self.assertIn("- Review status: reviewed", note)
        self.assertIn("| Alpha transport |", note)
        self.assertIn("twoPhaseSolver::alphaPredictor()", note)
        self.assertIn("alphaPredictor.C", note)
        self.assertIn("DeviceMULES.*", note)
        self.assertIn("| Pressure bridge |", note)
        self.assertIn("PressureMatrixCache", note)

    def test_render_source_audit_note_defaults_to_draft(self) -> None:
        bundle = load_authority_bundle(repo_root())

        note = render_source_audit_note(
            bundle,
            touched_surfaces=["Alpha transport"],
        )

        self.assertIn("- Review status: draft", note)

    def test_unknown_semantic_surface_fails_fast(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(ValueError, "unknown semantic contract surfaces: Made up surface"):
            resolve_source_audit_surfaces(bundle, ["Made up surface"])

    def test_validate_source_audit_note_accepts_reviewed_note_for_touched_surfaces(self) -> None:
        bundle = load_authority_bundle(repo_root())
        note = render_source_audit_note(
            bundle,
            touched_surfaces=["Alpha transport", "Pressure corrector"],
            review_status="reviewed",
        )

        result = validate_source_audit_note(
            bundle,
            note_text=note,
            touched_surfaces=["Alpha transport", "Pressure corrector"],
        )

        self.assertEqual(
            [entry.contract_surface for entry in result],
            ["Alpha transport", "Pressure corrector"],
        )

    def test_validate_source_audit_note_rejects_missing_reviewed_status(self) -> None:
        bundle = load_authority_bundle(repo_root())
        note = render_source_audit_note(
            bundle,
            touched_surfaces=["Alpha transport"],
            review_status="draft",
        )

        with self.assertRaisesRegex(ValueError, "source-audit note must be marked reviewed"):
            validate_source_audit_note(
                bundle,
                note_text=note,
                touched_surfaces=["Alpha transport"],
            )

    def test_validate_source_audit_note_rejects_missing_surface_coverage(self) -> None:
        bundle = load_authority_bundle(repo_root())
        note = render_source_audit_note(
            bundle,
            touched_surfaces=["Alpha transport"],
            review_status="reviewed",
        )

        with self.assertRaisesRegex(
            ValueError,
            "source-audit note is missing required semantic surfaces: Pressure corrector",
        ):
            validate_source_audit_note(
                bundle,
                note_text=note,
                touched_surfaces=["Alpha transport", "Pressure corrector"],
            )

    def test_validate_source_audit_note_rejects_local_target_family_drift(self) -> None:
        bundle = load_authority_bundle(repo_root())
        note = render_source_audit_note(
            bundle,
            touched_surfaces=["Pressure bridge"],
            review_status="reviewed",
        ).replace("PressureMatrixCache", "WrongTarget")

        with self.assertRaisesRegex(
            ValueError,
            "source-audit note row for 'Pressure bridge' does not match frozen semantic mapping",
        ):
            validate_source_audit_note(
                bundle,
                note_text=note,
                touched_surfaces=["Pressure bridge"],
            )

    def test_validate_source_audit_note_allows_expanded_ownership_scope(self) -> None:
        bundle = load_authority_bundle(repo_root())
        note = render_source_audit_note(
            bundle,
            touched_surfaces=["Alpha transport"],
            review_status="reviewed",
        )
        original_scope = (
            "local `alphaPredictor.C` path plus `DeviceAlphaTransport.*` / `DeviceMULES.*`"
        )
        expanded_scope = original_scope + " and ownership-reviewed Phase 5 boundaries"
        note = note.replace(
            f"{original_scope} | {original_scope}",
            f"{original_scope} | {expanded_scope}",
        )

        result = validate_source_audit_note(
            bundle,
            note_text=note,
            touched_surfaces=["Alpha transport"],
        )

        self.assertEqual(result[0].contract_surface, "Alpha transport")

    def test_validate_source_audit_note_accepts_alias_surface_labels_in_rows(self) -> None:
        bundle = load_authority_bundle(repo_root())
        note = render_source_audit_note(
            bundle,
            touched_surfaces=["alphaPredictor"],
            review_status="reviewed",
        ).replace("| Alpha transport |", "| alphaPredictor |")

        result = validate_source_audit_note(
            bundle,
            note_text=note,
            touched_surfaces=["alphaPredictor"],
        )

        self.assertEqual(result[0].contract_surface, "Alpha transport")

    def test_source_audit_cli_check_validates_reviewed_note(self) -> None:
        bundle = load_authority_bundle(repo_root())
        note = render_source_audit_note(
            bundle,
            touched_surfaces=["Alpha transport"],
            review_status="reviewed",
        )
        note_path = repo_root() / "tmp_source_audit_note.md"
        note_path.write_text(note, encoding="utf-8")
        self.addCleanup(note_path.unlink)

        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = source_audit.main(
                [
                    "check",
                    "--root",
                    str(repo_root()),
                    "--note",
                    str(note_path),
                    "--surface",
                    "Alpha transport",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Validated reviewed source-audit note", stdout.getvalue())

    def test_source_audit_cli_render_emits_standardized_note(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = source_audit.main(
                [
                    "render",
                    "--root",
                    str(repo_root()),
                    "--surface",
                    "Alpha transport",
                    "--surface",
                    "Pressure corrector",
                    "--title",
                    "Phase 5 Source Audit",
                ]
            )

        rendered = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("# Phase 5 Source Audit", rendered)
        self.assertIn("- Review status: draft", rendered)
        self.assertIn("## Semantic Surface Coverage", rendered)
        self.assertIn("| Alpha transport |", rendered)
        self.assertIn("| Pressure corrector |", rendered)

    def test_source_audit_script_entrypoint_runs_from_shell(self) -> None:
        completed = subprocess.run(
            [
                "python3",
                "scripts/authority/source_audit.py",
                "render",
                "--root",
                str(repo_root()),
                "--surface",
                "Alpha transport",
            ],
            cwd=repo_root(),
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("| Alpha transport |", completed.stdout)


if __name__ == "__main__":
    unittest.main()
