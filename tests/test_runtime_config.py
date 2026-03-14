from __future__ import annotations

import unittest

from scripts.symphony import runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_loads_default_codex_profiles(self) -> None:
        config = runtime_config.load_runtime_config()

        self.assertEqual(sorted(config.codex), ["implementation", "review"])
        self.assertEqual(config.codex["implementation"].model, "gpt-5.4")
        self.assertEqual(config.codex["implementation"].reasoning_effort, "medium")
        self.assertEqual(config.codex["review"].model, "gpt-5.4")
        self.assertEqual(config.codex["review"].reasoning_effort, "xhigh")

    def test_build_codex_command_uses_profile_settings(self) -> None:
        command = runtime_config.build_codex_command(
            "review",
            ["exec", "review", "--base", "origin/main"],
            codex_binary="/tmp/codex",
        )

        self.assertEqual(command[0], "/tmp/codex")
        self.assertIn("gpt-5.4", command)
        self.assertIn("model_reasoning_effort=xhigh", command)
        self.assertIn("shell_environment_policy.inherit=all", command)
        self.assertEqual(command[-3:], ["review", "--base", "origin/main"])


if __name__ == "__main__":
    unittest.main()
