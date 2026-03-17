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
    SOURCE_AUDIT_OWNERSHIP_SCOPE_PLACEHOLDER,
    validate_source_audit_note,
)
from scripts.authority import source_audit


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


class SourceAuditHelperTests(unittest.TestCase):
    @staticmethod
    def with_reviewed_ownership(note_text: str) -> str:
        return note_text.replace(
            SOURCE_AUDIT_OWNERSHIP_SCOPE_PLACEHOLDER,
            "ownership-reviewed scope boundary for this semantic surface",
        )

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
        self.assertEqual(
            resolved[0].ownership_scope, SOURCE_AUDIT_OWNERSHIP_SCOPE_PLACEHOLDER
        )
        self.assertEqual(
            resolved[1].ownership_scope, SOURCE_AUDIT_OWNERSHIP_SCOPE_PLACEHOLDER
        )

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
        self.assertIn(SOURCE_AUDIT_OWNERSHIP_SCOPE_PLACEHOLDER, note)

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
        note = self.with_reviewed_ownership(note)

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

    def test_validate_source_audit_note_rejects_placeholder_ownership_scope(self) -> None:
        bundle = load_authority_bundle(repo_root())
        note = render_source_audit_note(
            bundle,
            touched_surfaces=["Alpha transport"],
            review_status="reviewed",
        )

        with self.assertRaisesRegex(
            ValueError,
            "source-audit note row for 'Alpha transport' must record a reviewed ownership boundary",
        ):
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
        )
        note = self.with_reviewed_ownership(note).replace("PressureMatrixCache", "WrongTarget")

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
        original_scope = SOURCE_AUDIT_OWNERSHIP_SCOPE_PLACEHOLDER
        expanded_scope = original_scope + " and ownership-reviewed Phase 5 boundaries"
        note = note.replace(f"| {original_scope} |", f"| {expanded_scope} |")

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
        )
        note = self.with_reviewed_ownership(note).replace(
            "| Alpha transport |", "| alphaPredictor |"
        )

        result = validate_source_audit_note(
            bundle,
            note_text=note,
            touched_surfaces=["alphaPredictor"],
        )

        self.assertEqual(result[0].contract_surface, "Alpha transport")

    def test_validate_source_audit_note_rejects_duplicate_canonical_surface_rows(self) -> None:
        bundle = load_authority_bundle(repo_root())
        note = render_source_audit_note(
            bundle,
            touched_surfaces=["alphaPredictor"],
            review_status="reviewed",
        )
        duplicate_row = (
            "| alphaPredictor | `twoPhaseSolver::alphaPredictor()` | "
            "local `alphaPredictor.C` path plus `DeviceAlphaTransport.*` / `DeviceMULES.*` | "
            "ownership-reviewed duplicate | duplicate row |"
        )
        note = self.with_reviewed_ownership(note).replace(
            "## Reviewer Notes", duplicate_row + "\n\n## Reviewer Notes"
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicate semantic surface rows for 'Alpha transport'",
        ):
            validate_source_audit_note(
                bundle,
                note_text=note,
                touched_surfaces=["alphaPredictor"],
            )

    def test_source_audit_cli_check_validates_reviewed_note(self) -> None:
        bundle = load_authority_bundle(repo_root())
        note = render_source_audit_note(
            bundle,
            touched_surfaces=["Alpha transport"],
            review_status="reviewed",
        )
        note = self.with_reviewed_ownership(note)
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
