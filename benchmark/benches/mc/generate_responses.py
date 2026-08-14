# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MultiChallenge phase-A: generate responses against any OpenAI-compatible endpoint.

The official harness can't point the model-under-test and the judge at different
endpoints, so we generate ourselves (this script) and judge with the official
main.py --responses-file mode. Env: MC_BASE_URL, MC_API_KEY, MC_MODEL, MC_OUT,
MC_WORKERS (default 8), MC_QUESTIONS (default data/benchmark_questions.jsonl),
MC_EXTRA_BODY (optional JSON object merged into every request payload, e.g.
sampling params: '{"temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.5}').
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.environ["MC_BASE_URL"].rstrip("/")
KEY = os.environ["MC_API_KEY"]
MODEL = os.environ["MC_MODEL"]
OUT = os.environ["MC_OUT"]
WORKERS = int(os.environ.get("MC_WORKERS", "8"))
QUESTIONS = os.environ.get("MC_QUESTIONS", "data/benchmark_questions.jsonl")
EXTRA_BODY = json.loads(os.environ.get("MC_EXTRA_BODY", "{}"))


def _post(payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def call(messages: list[dict]) -> str:
    # no temperature (bedrock-Opus path rejects it); thinking disabled so the
    # weak tier's reasoning cannot starve the content budget (fallback drops
    # the knob for upstreams that reject it, e.g. Anthropic-format strong).
    payload = {"model": MODEL, "max_tokens": 8192, "messages": messages,
               "chat_template_kwargs": {"enable_thinking": False}, **EXTRA_BODY}
    last = None
    for attempt in range(3):
        try:
            body = _post(payload)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "chat_template_kwargs" in payload:
                payload.pop("chat_template_kwargs")
                continue
            last = e
            time.sleep(5 * (attempt + 1))
            continue
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last = e
            time.sleep(5 * (attempt + 1))
            continue
        content = body["choices"][0]["message"]["content"] or ""
        if content.strip():
            return content
        last = RuntimeError("empty content")
        payload.pop("chat_template_kwargs", None)
    raise RuntimeError(f"3 attempts failed: {last!r}")


def one(q: dict) -> dict:
    text = call(q["CONVERSATION"])
    return {"QUESTION_ID": q["QUESTION_ID"], "RESPONSE": [text]}


def main() -> None:
    qs = [json.loads(l) for l in open(QUESTIONS) if l.strip()]
    print(f"{len(qs)} questions -> {OUT} via {MODEL} @ {BASE} ({WORKERS} workers)")
    results, failed = {}, []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, q): q["QUESTION_ID"] for q in qs}
        done = 0
        for f in as_completed(futs):
            qid = futs[f]
            try:
                results[qid] = f.result()
            except Exception as e:  # noqa: BLE001
                failed.append(qid)
                print(f"FAILED {qid}: {e!r}", file=sys.stderr)
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(qs)} done ({len(failed)} failed)")
    with open(OUT, "w") as fh:
        for q in qs:  # preserve original order
            if q["QUESTION_ID"] in results:
                fh.write(json.dumps(results[q["QUESTION_ID"]]) + "\n")
    print(f"WROTE {len(results)}/{len(qs)} responses to {OUT}; failed: {len(failed)}")


if __name__ == "__main__":
    main()
