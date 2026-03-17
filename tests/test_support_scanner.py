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


if __name__ == "__main__":
    unittest.main()
