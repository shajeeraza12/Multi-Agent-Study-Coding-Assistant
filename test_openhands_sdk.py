"""
test_openhands_sdk.py - Standalone diagnostic for openhands-sdk + Ollama.

Bypasses graph.py, run_benchmark.py, and the entire ABC harness. Spins up a
minimal OpenHands Agent pointed at local Ollama and asks it to do one tiny
thing. Prints every step so you can see exactly where it fails.

Use this BEFORE running run_benchmark.py with --openhands-backend sdk. If
this script works, the SDK<->Ollama path is healthy. If it fails, the bug is
in (a) Ollama config, (b) tool-calling capability of the model, or (c) the
SDK setup - all isolated from the rest of the project.

Usage:
    python test_openhands_sdk.py
    python test_openhands_sdk.py --model ollama_chat/qwen2.5-coder:7b
    python test_openhands_sdk.py --max-iter 5 --prompt "List files in cwd"

Exit codes:
    0 = success (agent ran, got a final answer)
    1 = SDK import or build error
    2 = Ollama unreachable
    3 = run() raised
    4 = run() finished but no final answer extracted
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))


DEFAULT_PROMPT = (
    "Create a file named hello.py in the current working directory containing "
    "exactly: print('hello from openhands'). Then run it and report what it "
    "printed. Stop as soon as you have run it once."
)


def ping_ollama(base_url: str, model: str) -> bool:
    """Verify Ollama is up and the requested model is loaded."""
    import httpx
    tags_url = base_url.rstrip("/") + "/api/tags"
    print(f"[1/5] Pinging Ollama at {tags_url}")
    try:
        r = httpx.get(tags_url, timeout=5.0)
    except Exception as e:
        print(f"  FAIL: cannot reach Ollama: {e}")
        return False
    if r.status_code != 200:
        print(f"  FAIL: HTTP {r.status_code}")
        return False
    installed = [m.get("name", "") for m in r.json().get("models", [])]
    # Strip the litellm prefix when checking against ollama's installed list
    bare_model = model.split("/", 1)[-1] if "/" in model else model
    if not any(bare_model == m or bare_model in m for m in installed):
        print(f"  FAIL: model {bare_model!r} not in installed list: {installed}")
        return False
    print(f"  OK: model {bare_model!r} is loaded.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenHands SDK + Ollama diagnostic")
    parser.add_argument(
        "--model",
        default=None,
        help="litellm model string. Default derives from MODEL_NAME env var.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Ollama base URL (no /v1 suffix). Default: derived from OLLAMA_BASE_URL.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="The single instruction sent to the agent.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="Hard cap on agent steps. Default 10 - keep tight for diagnostics.",
    )
    parser.add_argument(
        "--workspace",
        default="workspace/diagnostic",
        help="Where the agent does its file work.",
    )
    args = parser.parse_args()

    chat_model = os.environ.get("MODEL_NAME", "qwen2.5:7b")
    model = args.model or os.environ.get("OPENHANDS_MODEL_NAME") or f"ollama_chat/{chat_model}"

    # Only derive base_url from Ollama settings if actually using ollama_chat
    base_url = args.base_url or os.environ.get("OPENHANDS_BASE_URL")
    if not base_url and model.startswith("ollama_chat/"):
        chat_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        base_url = chat_base.rstrip("/").removesuffix("/v1")

    api_key = os.environ.get("OPENHANDS_API_KEY")
    workspace_path = Path(args.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    print(f"=== OpenHands SDK diagnostic ===")
    print(f"Model:     {model}")
    print(f"Base URL:  {base_url or '(provider default)'}")
    print(f"API key:   {'set' if api_key else '(env-derived or none)'}")
    print(f"Workspace: {workspace_path.resolve()}")
    print(f"Max iter:  {args.max_iter}")
    print(f"Prompt:    {args.prompt!r}")
    print()

    # 1. Ollama reachability — only when actually using ollama_chat
    if model.startswith("ollama_chat/"):
        if not ping_ollama(base_url, model):
            print("\nSTOP: fix Ollama before continuing.")
            return 2
    else:
        print(f"[1/5] Skipping Ollama ping (model={model} is not ollama_chat)")

    # 2. SDK imports
    print(f"[2/5] Importing openhands-sdk")
    try:
        from openhands.sdk import LLM, Agent, Conversation
        from openhands.sdk.conversation.response_utils import get_agent_final_response
        from openhands.sdk.event import ActionEvent
        from openhands.sdk.event.llm_convertible.observation import ObservationEvent
        from openhands.tools.preset.default import get_default_tools
        print("  OK: imports succeeded.")
    except Exception as e:
        print(f"  FAIL: import error: {e}")
        traceback.print_exc()
        return 1

    # 3. Build LLM + agent + conversation
    print(f"[3/5] Building LLM + Agent + Conversation")
    try:
        llm_kwargs = dict(
            model=model,
            usage_id="diagnostic",
            temperature=0.0,
            num_retries=2,
            timeout=300,
            # Disable reasoning/thinking-mode parameters - the SDK defaults
            # target Claude/GPT-4 with extended thinking, which Ollama 7B
            # models reject with: '{model} does not support thinking'.
            reasoning_effort=None,
            extended_thinking_budget=0,
            enable_encrypted_reasoning=False,
        )
        if base_url:
            llm_kwargs["base_url"] = base_url
            if model.startswith("ollama_chat/"):
                llm_kwargs["ollama_base_url"] = base_url
        if api_key:
            llm_kwargs["api_key"] = api_key

        llm = LLM(**llm_kwargs)
        tools = get_default_tools(enable_browser=False)
        agent = Agent(llm=llm, tools=tools)

        action_trace = []
        step = {"i": 0}

        def trace_cb(event):
            try:
                if isinstance(event, ActionEvent) and getattr(event, "source", "") == "agent":
                    step["i"] += 1
                    print(f"  [STEP {step['i']}] action: tool={event.tool_name}")
                    action_trace.append(("action", event.tool_name, str(event.action)[:200]))
                elif isinstance(event, ObservationEvent):
                    obs_str = str(getattr(event, "observation", ""))[:200]
                    print(f"  [STEP {step['i']}] obs ({event.tool_name}): {obs_str}")
                    action_trace.append(("observation", event.tool_name, obs_str))
            except Exception as cb_err:
                print(f"  callback error (non-fatal): {cb_err}")

        conversation = Conversation(
            agent=agent,
            workspace=workspace_path,
            callbacks=[trace_cb],
            max_iteration_per_run=args.max_iter,
            stuck_detection=True,
        )
        print("  OK: conversation ready.")
    except Exception as e:
        print(f"  FAIL: build error: {e}")
        traceback.print_exc()
        return 1

    # 4. Send + run
    print(f"[4/5] Sending prompt and running agent (this may take several minutes)...")
    t0 = time.time()
    try:
        conversation.send_message(args.prompt)
        conversation.run()
    except Exception as e:
        print(f"  FAIL: conversation.run() raised: {e}")
        traceback.print_exc()
        return 3
    elapsed = time.time() - t0
    print(f"  OK: run finished in {elapsed:.1f}s after {step['i']} agent steps.")

    # 5. Extract final answer
    print(f"[5/5] Extracting final answer")
    try:
        events = list(conversation.state.events)
    except Exception:
        events = []
    final = get_agent_final_response(events)

    print()
    print(f"=== RESULT ===")
    print(f"Wall time:   {elapsed:.1f}s")
    print(f"Steps taken: {step['i']}")
    print(f"Trace length: {len(action_trace)}")
    print(f"Final answer:")
    print("-" * 60)
    print(final or "(empty)")
    print("-" * 60)

    # Quick check: did the agent actually create the hello.py file?
    hello = workspace_path / "hello.py"
    if hello.exists():
        print(f"\n[FILE CHECK] {hello} exists, contents:")
        print(hello.read_text(encoding="utf-8"))
    else:
        print(f"\n[FILE CHECK] {hello} does NOT exist.")

    if not final:
        print("\nWARN: no final answer extracted. The agent may have stalled.")
        return 4

    print("\nPASS: SDK + Ollama path is healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
