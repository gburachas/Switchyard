<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ayush Harbor routing experiments

This is the maintained native-TOML version of the Ayush routing sweep bundle. The old bundle used
YAML and `--routing-profiles`; the current Harbor runner accepts `--server-config` and starts the
Rust server itself.

The profiles are in `routing_configs/`:

- `llm-classifier-sticky-opus-qwen.toml` — capability classifier with sticky routing.
- `llm-classifier-perturn-opus-qwen.toml` — capability classifier on every request.
- `llm-classifier-sticky-opus-qwen-pertask.toml` — custom per-task selector.
- `single-qwen.toml` — weak-tier passthrough control.
- `stage-router-opus-qwen.toml` — signal-driven stage-router arm.

Prepare a TB-Lite or Terminal-Bench 2 Harbor dataset as described in `benchmark/README.md`, export
`OPENROUTER_API_KEY`, then run a profile through the generic launcher:

```bash
HARBOR_PATH=benchmark/datasets/openthoughts-tblite-closed-book \
bash benchmark/benches/experiments/ayush/run-harbor.sh \
  llm-classifier-sticky-opus-qwen.toml --agent claude-code --n-tasks 1
```

`run-tb-lite.sh` and `run-tb2.sh` supply the conventional generated-dataset paths. All remaining
arguments are forwarded to `benchmark/run-baseline.sh`, so a sweep can vary task list, concurrency,
agent, retries, output directory, and reasoning effort without maintaining duplicate wrappers.

The former `min_confidence` setting was removed from the two capability configs because it is not
part of the current native server schema. Their remaining routing semantics are preserved.
