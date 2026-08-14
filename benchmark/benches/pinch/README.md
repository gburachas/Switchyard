<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PinchBench profiles

These are native Switchyard operating points for an external PinchBench checkout. The benchmark
harness, task data, search credentials, judges, and run artifacts are deliberately not included.

Start the profile selected for a run, then configure the external harness to request its route id:

```bash
switchyard-server --config pinch-conservative-profile.toml --port 4046
```

`pinch-ctrl-allstrong.toml` is the strong-tier control. `pinch-v2-profile.toml`,
`pinch-v3-profile.toml`, and `pinch-conservative-profile.toml` are historical routing operating
points retained for comparative sweeps. They use the portable OpenRouter defaults; replace the
endpoint, credential variable, and model ids together for another provider.
