<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MultiChallenge profiles

This lane contains native Switchyard profiles and the small response-generation adapter needed
when the model-under-test and the official judge use different endpoints. It does not contain the
MultiChallenge harness, questions, labels, or generated responses: those materials are
internal-use-only.

Point `generate_responses.py` at a supplied questions JSONL and a running Switchyard route, then
give its response file to the official harness's response-file scoring mode. The profile route id
is normally `switchyard`:

```bash
switchyard-server --config mc-balanced-profile.toml --port 4046
MC_BASE_URL=http://127.0.0.1:4046/v1 \
MC_API_KEY=proxy MC_MODEL=switchyard MC_OUT=responses.jsonl \
MC_QUESTIONS=/path/to/benchmark_questions.jsonl \
python3 generate_responses.py
```

Do not commit the supplied questions, responses, or scored output.
