# Development Consensus

## Goal

Establish a concrete working agreement before implementation so code changes, tests, and process records stay consistent in this workspace.

## Changes Made

- Inspected the workspace structure and confirmed that the actual Git repository is `/home/wuhang/wuhang/linear_wh/triton`.
- Checked the existing archive-related directories and found historical empty folders `code-building/` and `codex-coding/`, while `docx/` is already in use.
- Created the standard directories `code_building/process_docs/`, `codex_coding/src/`, and `codex_coding/results/` for future work.
- Set the working agreement for this workspace:
  - Code changes default to `triton/`, and Git commands should run against the repository root in `triton/`.
  - New durable process records use `code_building/`; do not add new records to `code-building/`.
  - New helper scripts and test scripts use `codex_coding/src/`; execution outputs and logs use `codex_coding/results/`.
  - Important archived documents continue to use `docx/`.
  - Existing hyphenated directories remain untouched unless a dedicated migration task is requested.
  - Code comments must be written in English when comments are needed.
  - Relevant code, configuration, and documentation must be inspected before substantive edits.
  - After each meaningful milestone, update a process document under `code_building/process_docs/` and append a delta to `code_building/progress_diff_summary.md`.
  - For Triton work, follow `triton/AGENTS.md`, including rebuilding with `make` before tests and using focused pytest or lit commands.

## Verification

- Confirmed the workspace root with `pwd`.
- Confirmed the repository root with `git -C triton rev-parse --show-toplevel`.
- Confirmed the Triton worktree is clean with `git -C triton status --short`.
- Read `triton/AGENTS.md` to align local workflow with repository-specific testing guidance.

## Result

The development baseline is now defined. Future implementation can proceed with a stable convention for repository scope, archive paths, test artifact placement, and progress recording.

## Next Steps

- Start the next concrete development task inside `triton/` unless you specify another target.
- Record each meaningful implementation or verification round under `code_building/process_docs/`.
- Append each milestone delta to `code_building/progress_diff_summary.md`.

## 本轮命令

- `pwd`
- `ls -la`
- `rg --files -g 'code_building/**' -g 'codex_coding/**' -g 'docx/**'`
- `find code-building -maxdepth 3 -type f | sort`
- `find codex-coding -maxdepth 3 -type f | sort`
- `git status --short`
- `ls -la triton`
- `git -C triton rev-parse --show-toplevel`
- `find triton -maxdepth 2 -name .git -print`
- `sed -n '1,220p' AGENTS.md`
- `git -C /home/wuhang/wuhang/linear_wh/triton status --short`
- `mkdir -p code_building/process_docs codex_coding/src codex_coding/results`
