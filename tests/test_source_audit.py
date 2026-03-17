from __future__ import annotations

import pathlib
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
        self.assertEqual(resolved[0].ownership_scope, "alphaPredictor.C")
        self.assertEqual(resolved[1].ownership_scope, "pressureCorrector.C")

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
        self.assertIn("| Pressure bridge |", note)
        self.assertIn("PressureMatrixCache", note)

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
        self.assertIn("## Semantic Surface Coverage", rendered)
        self.assertIn("| Alpha transport |", rendered)
        self.assertIn("| Pressure corrector |", rendered)


if __name__ == "__main__":
    unittest.main()
