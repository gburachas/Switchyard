# Tau2 through the native Switchyard router

Install (py3.13 needs the audioop backport):
    git clone https://github.com/sierra-research/tau2-bench
    git -C tau2-bench checkout 1901a301961cbbe3fd11f3e84a2a376530c759e3
    python3 -m venv tau2-bench/.venv
    tau2-bench/.venv/bin/pip install -e ./tau2-bench  # add audioop-lts only on Python 3.13

Serve the router (cloud-only example; the lane's native TOMLs are alongside this guide):
    switchyard-server --config phase0-profile.toml --port 4048

Run (agent through the proxy; user simulator direct to the gateway):
    export OPENAI_API_KEY="$OPENROUTER_API_KEY"     # litellm reads this for the user-sim
    tau2 run --domain mock --num-trials 1 --num-tasks 1 --max-concurrency 1 \
      --agent-llm "openai/tau2" \
      --agent-llm-args '{"api_base":"http://127.0.0.1:4048/v1","api_key":"proxy"}' \
      --user-llm "openai/deepseek/deepseek-v4-flash" \
      --user-llm-args '{"api_base":"https://openrouter.ai/api/v1","extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'

Phase-0 result: mock task completed through the proxy — write actions 1/1, DB match 100%,
normal stop, 0 agent/user errors. Notes:
- `--task-ids` exists -> stratified subsets supported (plan open-question #4: YES).
- `telecom_small` is a ready-made small task set for iteration.
- Spark rule: any profile with the local weak tier runs at --max-concurrency 1 and NEVER
  while a benchmark owns :8001.
- The user simulator must set `enable_thinking:false`, otherwise Hard conversations can be dropped.
- `tau2-*.toml` profiles are native server configurations. `session_affinity = true` is the
  sticky control; an A/B that reclassifies on every user turn must use the native `followup`
  table rather than the retired `aow-serve --refresh` launcher.
