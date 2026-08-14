#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run one native Ayush routing configuration through the maintained Harbor launcher.
# Usage: HARBOR_PATH=/path/to/dataset bash run-harbor.sh CONFIG.toml [run-baseline options...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
CONFIG_NAME="${1:?usage: run-harbor.sh CONFIG.toml [run-baseline options...]}"
shift
CONFIG="$HERE/routing_configs/$CONFIG_NAME"

[[ -f "$CONFIG" ]] || { echo "Ayush config not found: $CONFIG" >&2; exit 1; }
: "${HARBOR_PATH:?set HARBOR_PATH to a generated Harbor dataset}"

exec bash "$REPO_ROOT/benchmark/run-baseline.sh" \
    --harbor-path "$HARBOR_PATH" \
    --server-config "$CONFIG" \
    --model "${ROUTE_MODEL:-switchyard}" \
    --agent "${AGENT:-claude-code}" \
    --n-concurrent "${N_CONCURRENT:-10}" \
    --max-retries "${MAX_RETRIES:-0}" \
    "$@"
