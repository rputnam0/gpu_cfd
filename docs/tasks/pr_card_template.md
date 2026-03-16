# PR Card Template

Use this structure for every PR card in the section docs.

```md
## <PR-ID> <Title>

- Objective:
- Exact citations:
  - Authority:
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
  - Backlog scope:
  - Backlog done_when:
- Depends on (backlog IDs):
- Prerequisites:
- Concrete task slices:
  1.
  2.
  3.
- Artifacts/contracts introduced or consumed:
- Validation:
- Done criteria:
- Exports to later PRs/phases:
```

## Section Trailer

Every section doc must end with these blocks, using these exact names:

```md
## imports_from_prev

- <contract/artifact>

## exports_to_next

- <contract/artifact>

## shared_terms

- `<term>`: <canonical meaning used in this section>

## open_discontinuities

- `[blocking|tracked] <name>`: <issue, citations, impacted PR IDs, preferred reading>

## validation_checks

- <section-level review gate>
```

## Required Practices

- Cite exact subsection anchors from the phase spec, not only the document name.
- `FND-*` cards may use the relevant [authority/README.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/authority/README.md) authority-order subsection in place of a phase-spec citation.
- Preserve every `depends_on` edge from the backlog.
- Break work into implementation-ready task slices, not vague themes.
- Prefer exported artifact names already frozen in the authority docs when they exist.
