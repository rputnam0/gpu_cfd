from __future__ import annotations

import unittest

from scripts.symphony import runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_loads_default_codex_profiles(self) -> None:
        config = runtime_config.load_runtime_config()

        self.assertEqual(sorted(config.codex), ["implementation", "review"])
        self.assertEqual(config.codex["implementation"].model, "gpt-5.4")
        self.assertEqual(config.codex["implementation"].reasoning_effort, "high")
        self.assertIn(
            "features.multi_agent=true",
            config.codex["implementation"].extra_configs,
        )
        self.assertIn(
            "features.child_agents_md=true",
            config.codex["implementation"].extra_configs,
        )
        self.assertIn(
            "agents.max_threads=3",
            config.codex["implementation"].extra_configs,
        )
        self.assertIn(
            "agents.max_depth=1",
            config.codex["implementation"].extra_configs,
        )
        self.assertIn(
            "agents.job_max_runtime_seconds=900",
            config.codex["implementation"].extra_configs,
        )
        self.assertEqual(config.codex["review"].model, "gpt-5.4")
        self.assertEqual(config.codex["review"].reasoning_effort, "xhigh")
        self.assertEqual(config.codex["review"].timeout_seconds, 900)

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

    def test_build_implementation_command_enables_multi_agent_support(self) -> None:
        command = runtime_config.build_codex_command(
            "implementation",
            ["app-server"],
            codex_binary="/tmp/codex",
        )

        self.assertIn("features.multi_agent=true", command)
        self.assertIn("features.child_agents_md=true", command)
        self.assertIn("agents.max_threads=3", command)
        self.assertIn("agents.max_depth=1", command)
        self.assertIn("agents.job_max_runtime_seconds=900", command)


if __name__ == "__main__":
    unittest.main()
