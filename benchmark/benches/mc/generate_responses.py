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
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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


def _write_responses(output_path: Path, questions: list[dict], results: dict) -> None:
    """Atomically replace output_path once every response has succeeded."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for question in questions:  # preserve original order
                output_file.write(json.dumps(results[question["QUESTION_ID"]]) + "\n")
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    with open(QUESTIONS, encoding="utf-8") as questions_file:
        qs = [json.loads(line) for line in questions_file if line.strip()]
    print(f"{len(qs)} questions -> {OUT} via {MODEL} @ {BASE} ({WORKERS} workers)")
    results, failed = {}, []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, q): q["QUESTION_ID"] for q in qs}
        done = 0
        for future in as_completed(futs):
            qid = futs[future]
            try:
                results[qid] = future.result()
            except Exception as exc:  # noqa: BLE001
                failed.append(qid)
                print(f"FAILED {qid}: {exc!r}", file=sys.stderr)
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(qs)} done ({len(failed)} failed)")
    if failed:
        print(
            f"FAILED {len(failed)}/{len(qs)} responses; no output was written to {OUT}",
            file=sys.stderr,
        )
        return 1

    _write_responses(Path(OUT), qs, results)
    print(f"WROTE {len(results)}/{len(qs)} responses to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
