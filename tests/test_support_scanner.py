from __future__ import annotations

import unittest

from scripts.authority import (
    SupportBoundaryCondition,
    SupportFunctionObject,
    SupportScanRequest,
    SupportScanRejected,
    SupportStartupSeedFieldValue,
    SupportStartupSeedRegion,
    SupportStartupSeedSpec,
    enforce_support_scan,
    load_authority_bundle,
    repo_root,
    scan_support_matrix,
)


class SupportScannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_authority_bundle(repo_root())

    def make_supported_request(self) -> SupportScanRequest:
        return SupportScanRequest(
            execution_mode="production",
            fallback_policy="failFast",
            backend="native",
            mesh_mode="static",
            region_count=1,
            processor_patches_present=False,
            cyclic_or_ami_patches_present=False,
            arbitrary_coded_patch_fields_present=False,
            turbulence_model="laminar",
            contact_angle_enabled=False,
            surface_tension_model="constant_sigma",
            schemes={
                "ddtSchemes": {"default": "Euler"},
                "gradSchemes": {
                    "default": "Gauss linear",
                    "grad(alpha.water)": "Gauss linear",
                },
                "interpolationSchemes": {
                    "default": "linear",
                    "interpolate(grad(alpha.water))": "linear",
                    "interpolate(sigmaK)": "linear",
                },
                "divSchemes": {
                    "div(phi,alpha.water)": "Gauss upwind",
                    "div(phirb,alpha.water)": "Gauss interfaceCompression",
                },
            },
            function_objects=(
                SupportFunctionObject(
                    name="min_alpha",
                    class_name="fieldMinMax",
                    execute_control="writeTime",
                ),
            ),
            boundary_conditions=(
                SupportBoundaryCondition(
                    patch_role="Swirl inlet (swirlInletA/B)",
                    field="U",
                    kind="gpuPressureSwirlInletVelocity",
                ),
                SupportBoundaryCondition(
                    patch_role="Swirl inlet (swirlInletA/B)",
                    field="p_rgh",
                    kind="prghTotalHydrostaticPressure",
                ),
                SupportBoundaryCondition(
                    patch_role="Swirl inlet (swirlInletA/B)",
                    field="alpha.water",
                    kind="fixedValue",
                ),
                SupportBoundaryCondition(
                    patch_role="Wall",
                    field="U",
                    kind="noSlip",
                ),
                SupportBoundaryCondition(
                    patch_role="Wall",
                    field="p_rgh",
                    kind="fixedFluxPressure",
                ),
                SupportBoundaryCondition(
                    patch_role="Wall",
                    field="alpha.water",
                    kind="zeroGradient",
                ),
                SupportBoundaryCondition(
                    patch_role="Ambient/open",
                    field="U",
                    kind="pressureInletOutletVelocity",
                ),
                SupportBoundaryCondition(
                    patch_role="Ambient/open",
                    field="p_rgh",
                    kind="prghPressure",
                ),
                SupportBoundaryCondition(
                    patch_role="Ambient/open",
                    field="alpha.water",
                    kind="inletOutlet",
                ),
            ),
            startup_seed=SupportStartupSeedSpec(
                enabled=True,
                force_reseed=False,
                precedence="lastWins",
                default_field_values=(
                    SupportStartupSeedFieldValue(
                        value_class="volScalarFieldValue",
                        field="alpha.water",
                        value=1.0,
                    ),
                ),
                regions=(
                    SupportStartupSeedRegion(
                        region_type="cylinderToCell",
                        field_values=(
                            SupportStartupSeedFieldValue(
                                value_class="volVectorFieldValue",
                                field="U",
                                value="(0 0 1)",
                            ),
                        ),
                    ),
                ),
            ),
        )

    def test_supported_production_request_passes_with_clean_report(self) -> None:
        report = scan_support_matrix(self.bundle, self.make_supported_request())

        self.assertTrue(report.startup_allowed)
        self.assertTrue(report.production_eligible)
        self.assertEqual(report.mode_label, "production")
        self.assertEqual(report.fallback_policy, "failFast")
        self.assertEqual(report.reject_reasons, ())
        self.assertEqual(report.as_dict()["authority_citations"][0], "docs/authority/support_matrix.json#global_policy")

    def test_production_rejects_unsupported_configurations_before_startup(self) -> None:
        request = self.make_supported_request()
        request = SupportScanRequest(
            **{
                **request.__dict__,
                "backend": "amgx",
                "backend_pressure_mode": "PinnedHost",
                "turbulence_model": "kOmegaSST",
                "schemes": {
                    **request.schemes,
                    "ddtSchemes": {"default": "localEuler"},
                },
                "function_objects": (
                    SupportFunctionObject(
                        name="debug_residuals",
                        class_name="residuals",
                        execute_control="timeStep",
                    ),
                ),
                "boundary_conditions": (
                    SupportBoundaryCondition(
                        patch_role="Wall",
                        field="U",
                        kind="slip",
                    ),
                ),
                "startup_seed": SupportStartupSeedSpec(
                    enabled=True,
                    force_reseed=False,
                    precedence="firstWins",
                    default_field_values=(
                        SupportStartupSeedFieldValue(
                            value_class="volScalarFieldValue",
                            field="alpha.water",
                            value=1.0,
                        ),
                    ),
                    regions=(),
                    extra_keys=("unsupportedKey",),
                ),
            }
        )

        with self.assertRaises(SupportScanRejected) as context:
            enforce_support_scan(self.bundle, request)

        report = context.exception.report
        codes = tuple(reason["code"] for reason in report.reject_reasons)
        self.assertEqual(
            codes,
            (
                "backend_mode_not_admitted",
                "function_object_debug_only_in_production",
                "missing_boundary_condition",
                "startup_seed_precedence_not_admitted",
                "startup_seed_unknown_key",
                "turbulence_scope_violation",
                "unsupported_boundary_condition",
                "unsupported_scheme_tuple",
            ),
        )
        self.assertFalse(report.startup_allowed)
        self.assertTrue(all(reason["citations"] for reason in report.reject_reasons))

    def test_debug_only_fallback_requires_opt_in_and_sets_mode_label(self) -> None:
        request = self.make_supported_request()
        unsupported_request = SupportScanRequest(
            **{
                **request.__dict__,
                "execution_mode": "debug",
                "function_objects": (
                    SupportFunctionObject(
                        name="solver_info",
                        class_name="solverInfo",
                        execute_control="timeStep",
                    ),
                ),
            }
        )

        with self.assertRaises(SupportScanRejected):
            enforce_support_scan(self.bundle, unsupported_request)

        fallback_request = SupportScanRequest(
            **{
                **unsupported_request.__dict__,
                "fallback_policy": "debugOnlyFallback",
            }
        )
        report = enforce_support_scan(self.bundle, fallback_request)

        self.assertTrue(report.startup_allowed)
        self.assertFalse(report.production_eligible)
        self.assertEqual(report.mode_label, "debug-only-fallback")
        self.assertEqual(
            tuple(reason["code"] for reason in report.reject_reasons),
            ("function_object_debug_only_enabled",),
        )

    def test_unsupported_backend_names_fail_fast(self) -> None:
        request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "backend": "petsc",
            }
        )

        with self.assertRaises(SupportScanRejected) as context:
            enforce_support_scan(self.bundle, request)

        self.assertEqual(
            tuple(reason["code"] for reason in context.exception.report.reject_reasons),
            ("backend_not_admitted",),
        )

    def test_production_debug_fallback_policy_is_not_production_eligible(self) -> None:
        request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "fallback_policy": "debugOnlyFallback",
            }
        )

        with self.assertRaises(SupportScanRejected) as context:
            enforce_support_scan(self.bundle, request)

        report = context.exception.report
        self.assertFalse(report.startup_allowed)
        self.assertFalse(report.production_eligible)
        self.assertEqual(
            tuple(reason["code"] for reason in report.reject_reasons),
            ("production_debug_fallback_forbidden",),
        )

    def test_symmetry_and_empty_boundaries_are_admitted(self) -> None:
        request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "boundary_conditions": self.make_supported_request().boundary_conditions
                + (
                    SupportBoundaryCondition(
                        patch_role="Symmetry / empty",
                        field="U",
                        kind="pass-through",
                    ),
                    SupportBoundaryCondition(
                        patch_role="Symmetry / empty",
                        field="alpha.water",
                        kind="no-op semantics",
                    ),
                ),
            }
        )

        report = enforce_support_scan(self.bundle, request)

        self.assertTrue(report.startup_allowed)
        self.assertEqual(report.reject_reasons, ())

    def test_startup_seed_rejects_malformed_scalar_and_vector_values(self) -> None:
        request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "startup_seed": SupportStartupSeedSpec(
                    enabled=True,
                    force_reseed=False,
                    precedence="lastWins",
                    default_field_values=(
                        SupportStartupSeedFieldValue(
                            value_class="volScalarFieldValue",
                            field="alpha.water",
                            value="(0 0 1)",
                        ),
                    ),
                    regions=(
                        SupportStartupSeedRegion(
                            region_type="cylinderToCell",
                            field_values=(
                                SupportStartupSeedFieldValue(
                                    value_class="volVectorFieldValue",
                                    field="U",
                                    value=1.0,
                                ),
                            ),
                        ),
                    ),
                ),
            }
        )

        with self.assertRaises(SupportScanRejected) as context:
            enforce_support_scan(self.bundle, request)

        self.assertEqual(
            tuple(reason["code"] for reason in context.exception.report.reject_reasons),
            (
                "startup_seed_field_value_type_not_admitted",
                "startup_seed_field_value_type_not_admitted",
            ),
        )

    def test_extra_alpha_air_scheme_entries_are_rejected(self) -> None:
        request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "schemes": {
                    **self.make_supported_request().schemes,
                    "gradSchemes": {
                        **self.make_supported_request().schemes["gradSchemes"],
                        "grad(alpha.air)": "Gauss linear",
                    },
                },
            }
        )

        with self.assertRaises(SupportScanRejected) as context:
            enforce_support_scan(self.bundle, request)

        self.assertEqual(
            tuple(reason["code"] for reason in context.exception.report.reject_reasons),
            ("unsupported_scheme_tuple",),
        )

    def test_debug_amgx_rejects_unknown_bridge_modes(self) -> None:
        request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "execution_mode": "debug",
                "backend": "amgx",
                "backend_pressure_mode": "TypoMode",
            }
        )

        with self.assertRaises(SupportScanRejected) as context:
            enforce_support_scan(self.bundle, request)

        self.assertEqual(
            tuple(reason["code"] for reason in context.exception.report.reject_reasons),
            ("backend_mode_not_admitted",),
        )

    def test_extra_scheme_blocks_are_rejected(self) -> None:
        request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "schemes": {
                    **self.make_supported_request().schemes,
                    "snGradSchemes": {"default": "corrected"},
                },
            }
        )

        with self.assertRaises(SupportScanRejected) as context:
            enforce_support_scan(self.bundle, request)

        self.assertEqual(
            tuple(reason["code"] for reason in context.exception.report.reject_reasons),
            ("unsupported_scheme_tuple",),
        )

    def test_unknown_execution_and_fallback_modes_fail_fast(self) -> None:
        execution_request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "execution_mode": "async_no_graph",
            }
        )
        fallback_request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "fallback_policy": "typo",
            }
        )

        with self.assertRaises(SupportScanRejected) as execution_context:
            enforce_support_scan(self.bundle, execution_request)
        with self.assertRaises(SupportScanRejected) as fallback_context:
            enforce_support_scan(self.bundle, fallback_request)

        self.assertEqual(
            tuple(reason["code"] for reason in execution_context.exception.report.reject_reasons),
            ("execution_mode_not_admitted",),
        )
        self.assertEqual(
            tuple(reason["code"] for reason in fallback_context.exception.report.reject_reasons),
            ("fallback_policy_not_admitted",),
        )

    def test_duplicate_alpha_alias_scheme_entries_are_rejected(self) -> None:
        request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "schemes": {
                    **self.make_supported_request().schemes,
                    "gradSchemes": {
                        **self.make_supported_request().schemes["gradSchemes"],
                        "grad(alpha1)": "Gauss linear",
                    },
                },
            }
        )

        with self.assertRaises(SupportScanRejected) as context:
            enforce_support_scan(self.bundle, request)

        self.assertEqual(
            tuple(reason["code"] for reason in context.exception.report.reject_reasons),
            ("unsupported_scheme_tuple",),
        )

    def test_missing_required_nozzle_boundary_rows_are_rejected(self) -> None:
        request = SupportScanRequest(
            **{
                **self.make_supported_request().__dict__,
                "boundary_conditions": tuple(
                    condition
                    for condition in self.make_supported_request().boundary_conditions
                    if not (condition.patch_role == "Ambient/open" and condition.field == "alpha.water")
                ),
            }
        )

        with self.assertRaises(SupportScanRejected) as context:
            enforce_support_scan(self.bundle, request)

        self.assertEqual(
            tuple(reason["code"] for reason in context.exception.report.reject_reasons),
            ("missing_boundary_condition",),
        )


if __name__ == "__main__":
    unittest.main()
