from __future__ import annotations

import unittest

from scripts.symphony import preflight


class ParseMcpListTests(unittest.TestCase):
    def test_parses_linear_row(self) -> None:
        rows = preflight.parse_mcp_list(
            "Name    Url                         Bearer Token Env Var  Status   Auth\n"
            "linear  https://mcp.linear.app/mcp  -                     enabled  OAuth\n"
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "linear")
        self.assertEqual(rows[0]["status"], "enabled")
        self.assertEqual(rows[0]["auth"], "OAuth")

    def test_ignores_non_table_output(self) -> None:
        self.assertEqual(
            preflight.parse_mcp_list("No MCP servers configured yet.\n"), []
        )


if __name__ == "__main__":
    unittest.main()
