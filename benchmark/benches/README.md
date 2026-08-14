<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Routing benchmark profiles

Native Switchyard configurations and thin helpers for routing benchmarks that use external
harnesses. They measure the effect of selecting a weak or strong model tier without copying the
benchmark harness, restricted datasets, credentials, or run artifacts into this repository.

## Layout

```
tau2/              tau2-bench: sticky-vs-follow-up A/B and tier-flip profiles
pinch/             PinchBench routing operating points
mc/                MultiChallenge routing operating points and response generator
experiments/ayush/ Updated Ayush Harbor sweeps for TB-Lite and Terminal-Bench 2
```

## What you must supply

Nothing in this directory is runnable without the following inputs, which it deliberately does
not ship:

1. **A benchmark checkout and task data.** The profiles drive third-party harnesses
   ([tau2-bench](https://github.com/sierra-research/tau2-bench), PinchBench, MultiChallenge, and
   Harbor). Pin the harness version because scoring changes across releases.

   **No benchmark task content or answers are in this tree.** MultiChallenge source data is
   internal-use-only and must never be copied into this repository.
2. **Endpoints and a key.** The profiles use `https://openrouter.ai/api/v1` for the
   strong tier and `http://127.0.0.1:8001/v1` for the weak tier and classifier, which assumes a
   local llama.cpp or vLLM server. Substitute your own and export `OPENROUTER_API_KEY`. The model
   ids are placeholders (`local-weak`), not recommendations.
3. **Benchmark-specific evaluation and scoring.** Use the official harness for the benchmark being
   run. This repository provides routing configuration, not a replacement harness.

## Validate a profile

Build the native server, then validate a configuration before using it with a harness:

```bash
for f in benchmark/benches/{tau2,pinch,mc}/*.toml \
         benchmark/benches/experiments/ayush/routing_configs/*.toml; do
    ./target/release/switchyard-server --config "$f" --dry-run
done
```

The individual lane README files describe their required environment variables and harness entry
points. Keep run artifacts (`logs/`, `results/`, `*.log`) out of version control.
