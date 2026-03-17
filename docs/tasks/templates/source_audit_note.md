# Source Audit Note

Use this template for any reviewed semantic source-audit artifact derived from the Foundation
semantic-source helper. Keep the row values aligned with the frozen authority bundle; do not
redefine local target families in downstream phase notes.

## Review Status

- Review status: `reviewed|draft`
- Authority source: `docs/authority/semantic_source_map.md`
- Contract note: patch the local SPUMA/v2412 target family named by the authority bundle, not an upstream analog path.

## Semantic Surface Coverage

| Contract surface | Semantic reference | Local target family | Ownership scope | Notes |
| --- | --- | --- | --- | --- |
| `<surface>` | `<semantic reference>` | `<local target family>` | `<ownership scope>` | `<phase-local reviewer note or blocker>` |

## Reviewer Notes

- Record unresolved symbol drift as a blocker before implementation.
- Add any phase-local patch boundaries here without changing the frozen target-family mapping above.
