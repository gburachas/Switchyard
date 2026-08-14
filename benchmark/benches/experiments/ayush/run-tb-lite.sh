#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HARBOR_PATH="${HARBOR_PATH:-$HERE/../../../datasets/openthoughts-tblite-closed-book}"
exec bash "$HERE/run-harbor.sh" "$@"
