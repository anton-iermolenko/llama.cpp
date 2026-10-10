# Local llama-server fixes

The maintained Git source is `C:/Users/anton/Projects/forks/llama.cpp`.
Fork: https://github.com/anton-iermolenko/llama.cpp (`origin`).
Upstream: https://github.com/ggml-org/llama.cpp.git (`upstream`).
Base: llama.cpp b10809, `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`.
Migrated on 2026-10-09 from the existing local-gpu-llm patches.

## Branches and commits

These branches form a cumulative stack. Each branch includes the preceding fixes.
`codex/local-server-fixes` is the integration branch for future local work.

| Branch | Fix commit | Change |
| --- | --- | --- |
| `codex/mtp-bulk-read` | `c8ceadab2` | Copy dense MTP verification rows in bulk |
| `codex/mtp-gpu-handoff` | `83e59e587` | Transfer target hidden states to draft on CUDA |
| `codex/kv-rotation-cache` | `a8921e25d` | Retain immutable Qwen35 rotation constants on CUDA |
| `codex/mtp-embedding-reuse` | `825017346` | Reuse target token embeddings during draft catch-up |

## Working with the clone

```powershell
Set-Location C:\Users\anton\Projects\forks\llama.cpp
git switch codex/local-server-fixes
git log --oneline 5266f24da..HEAD
git diff 5266f24da..HEAD -- common src
```

The migration preserves existing implementation behavior and the pinned upstream
base. No upstream push, runtime replacement, model reload or benchmark rerun was
performed. The clone uses LF line endings (`core.autocrlf=false`).

## Build and validation provenance

Historical tooling and benchmark evidence remain in
`C:/Users/anton/Projects/local-gpu-llm`:

- Bulk reads: `scripts/mtp-bulk/`, `scripts/build-mtp-bulk.py`,
  `results/mtp-bulk-optimization/REPORT.md`.
- GPU handoff: `scripts/mtp-gpu/`, `scripts/build-mtp-gpu.py`,
  `results/mtp-gpu-handoff/REPORT.md`.
- Rotation cache: `scripts/rotation-cache/`, `scripts/build-rotation-cache.py`,
  `results/rotation-cache/REPORT.md`.
- Embedding reuse: `scripts/mtp-embedding-reuse/`, `scripts/build-mtp-embedding.py`,
  `results/mtp-embedding-reuse/REPORT.md`.

The old source directories under `runtime/llama-mtp-*-source` remain preserved
as build snapshots. Existing workspace builders and CMake caches still point
there; they do not automatically build this clone. Future builds from this clone
must use a new build directory and explicitly select this source tree. Do not
reuse the historical patch applicators on the clone: the patches are committed.

The active runtime remains `runtime/llama-b10809-mtp-embedding`. Its build record
is `results/mtp-embedding-reuse/build-report.json`. Existing evidence covers the
local Qwen35 setup; the migration does not establish support for other models or
backends.

The five local branches were published to the GitHub fork on 2026-10-10.
Each branch tracks its corresponding origin branch. No upstream PR was opened.
