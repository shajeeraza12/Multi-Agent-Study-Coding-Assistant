"""
openhands_agent.py - Condition C "Hallucination Bridge" entry point.

================================================================================
THREE-BACKEND DISPATCHER
================================================================================

Selects which OpenHands integration to drive based on the OPENHANDS_BACKEND
env var. Set it in .env, the shell, or programmatically before graph import.

  OPENHANDS_BACKEND=cloud    -> Existing OpenHands Cloud API (app.all-hands.dev).
                                 This is the production path and the default if
                                 OPENHANDS_BACKEND is not set. Slow (2-3 min per
                                 call), uses Cloud credits, requires internet.

  OPENHANDS_BACKEND=stub     -> Returns a hardcoded plausible answer with no
                                 network calls. Used for fast mock/wiring tests
                                 of the variant='c' graph topology and the
                                 Reviewer audit path. Sub-second per call.

  OPENHANDS_BACKEND=sdk      -> Local openhands-sdk runtime (Phase 2, not yet
                                 implemented). Will spawn a local Agent, execute
                                 terminal/file actions in a sandbox, and surface
                                 the per-action trace to the Reviewer Agent.
                                 Currently raises NotImplementedError with a
                                 pointer to the migration plan.

Public surface (unchanged from before this refactor):
    from openhands_agent import create_openhands_chain
    chain = create_openhands_chain()           # called once during graph build
    result = chain(state)                       # called per ChatState invocation
    # result == {"code_answer": "...", "openhands_status": "...", ...}

graph.py:build_graph(variant='c') imports `create_openhands_chain` and wires the
returned callable into the variant-C code_helper node. Nothing in graph.py
needs to change when we swap backends.

================================================================================
"""

import os
from typing import Dict, Callable, Any

from openhands_client import create_openhands_client


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _resolve_backend() -> str:
    """Read OPENHANDS_BACKEND env var, normalize to lowercase, default 'cloud'."""
    raw = (os.environ.get("OPENHANDS_BACKEND") or "cloud").strip().lower()
    if raw not in ("stub", "cloud", "sdk"):
        print(f"[OPENHANDS] Unknown OPENHANDS_BACKEND='{raw}', falling back to 'cloud'.")
        return "cloud"
    return raw


# ---------------------------------------------------------------------------
# Stub backend - hardcoded answer for fast wiring tests
# ---------------------------------------------------------------------------

# Plausible solution to the toy mock prompt. Picked so the Reviewer's
# 5-dimension rubric (Correctness, Edge Cases, Security, Quality, Relevance)
# scores it >= 0.7 on the first pass. This avoids triggering the refine loop,
# which is important because graph.py:code_node_c hardcodes iteration_count=1
# without incrementing - if Reviewer ever scored below 0.7 in stub mode, we
# would loop forever.
_STUB_MOCK_ANSWER = """\
Here is a Python implementation of `reverse_string`:

```python
def reverse_string(s: str) -> str:
    \"\"\"Return the reverse of the input string.

    Args:
        s: The string to reverse. Must be a str.

    Returns:
        The input string reversed. Empty string returns empty string.

    Examples:
        >>> reverse_string("hello")
        'olleh'
        >>> reverse_string("")
        ''
    \"\"\"
    if not isinstance(s, str):
        raise TypeError(f"reverse_string expected str, got {type(s).__name__}")
    return s[::-1]
```

This solution uses Python's slice-reverse idiom `s[::-1]` which is O(n) time
and space, handles empty strings correctly, and includes a runtime type guard.
The docstring documents the contract, edge cases, and provides doctest examples.
"""


def _invoke_stub(state: Dict) -> Dict:
    """Stub mode: return hardcoded answer, no network. Used for fast wiring tests."""
    question = state.get("code_question") or state.get("main_task") or ""
    print(f"[OPENHANDS-STUB] Backend=stub. Question (first 80): {question[:80]}")
    print(f"[OPENHANDS-STUB] swe_instance_id={state.get('swe_instance_id')}")
    print(f"[OPENHANDS-STUB] Returning hardcoded mock answer (no Cloud, no SDK).")
    return {
        "code_answer": _STUB_MOCK_ANSWER,
        "openhands_status": "success",
        "openhands_backend": "stub",
    }


# ---------------------------------------------------------------------------
# Cloud backend - existing OpenHands Cloud API path (was the only path before)
# ---------------------------------------------------------------------------

def _invoke_cloud(state: Dict, timeout: int = 120) -> Dict:
    """
    Cloud mode: hit OpenHands Cloud (app.all-hands.dev) via openhands_client.

    This is the original implementation; behavior is preserved exactly.
    """
    # Local import to avoid circular deps at module load
    from agents import sanitize_user_text

    user_prompt = state.get("code_question", "") or state.get("main_task", "")
    if not user_prompt:
        return {
            "code_answer": "No task provided",
            "openhands_status": "error",
            "openhands_backend": "cloud",
        }

    safe_prompt = sanitize_user_text(user_prompt)
    client = create_openhands_client(timeout=timeout)

    status = client.check_status()
    if status.get("status") != "running":
        return {
            "code_answer": f"OpenHands Cloud not reachable: {status.get('message', 'Unknown')}",
            "openhands_status": "error",
            "openhands_backend": "cloud",
        }

    try:
        result = client.execute_task(safe_prompt)

        if result.get("status") == "timeout":
            return {
                "code_answer": f"Timeout after {timeout}s",
                "openhands_status": "timeout",
                "openhands_backend": "cloud",
            }

        if result.get("status") == "error":
            return {
                "code_answer": f"Error: {result.get('message', 'Unknown')}",
                "openhands_status": "error",
                "openhands_backend": "cloud",
            }

        code_answer = result.get("output", "") or "No solution generated"
        return {
            "code_answer": code_answer,
            "openhands_status": "success",
            "openhands_backend": "cloud",
        }

    except Exception as e:
        return {
            "code_answer": f"Error: {str(e)}",
            "openhands_status": "error",
            "openhands_backend": "cloud",
        }


# ---------------------------------------------------------------------------
# SDK backend - Phase 2 placeholder
# ---------------------------------------------------------------------------

def _invoke_sdk(state: Dict) -> Dict:
    """
    SDK mode: drive a local openhands-sdk Agent via Ollama.

    What this does:
      1. Build an LLM pointed at local Ollama using litellm's ollama_chat
         provider format (model="ollama_chat/<MODEL_NAME>").
      2. Get the default tool set (terminal + file_editor + task_tracker).
         Browser tools are disabled (cli_mode style) for headless benchmarking.
      3. Construct an Agent.
      4. Create a per-instance LocalConversation with a workspace dir
         scoped to the swe_instance_id, so file edits don't collide across
         runs.
      5. Register a callback that captures every ActionEvent + ObservationEvent
         into a trace buffer. This is the raw input the Reviewer Agent uses
         for the "Hallucination Bridge" audit.
      6. send_message + run with a tight iteration cap (default 20) so a
         confused 7B model can't burn hours on one task.
      7. Extract the final answer with get_agent_final_response().
      8. Return code_answer + the action trace under "openhands_action_trace".

    Configuration via env vars (read at chain creation, not per-call):
      OPENHANDS_MODEL_NAME         - model string sent to litellm; default is
                                     "ollama_chat/<MODEL_NAME>" where MODEL_NAME
                                     is the same chat model used by Conditions
                                     A and B (preserves comparative rigor).
      OPENHANDS_BASE_URL           - Ollama server base URL (no /v1 suffix).
                                     Default: derived from OLLAMA_BASE_URL by
                                     stripping /v1.
      OPENHANDS_MAX_ITERATIONS     - cap on agent step count per run.
                                     Default: 20.
      OPENHANDS_WORKSPACE_ROOT     - directory under which per-instance
                                     workspaces are created.
                                     Default: ./workspace
      OPENHANDS_DISABLE_BROWSER    - default "1" (browser disabled). Set to
                                     "0" to enable browser tool.

    Returns:
      A dict at minimum containing "code_answer", "openhands_status",
      "openhands_backend", and (on success) "openhands_action_trace" -
      a list of {"step", "type", "tool_name", "action_summary",
      "observation_summary"} dicts in chronological order.
    """
    import time
    import traceback
    from pathlib import Path

    # Lazy imports so the module loads even if openhands-sdk isn't installed
    # (e.g., on py3.10 fallback environments).
    try:
        from openhands.sdk import (
            LLM,
            Agent,
            Conversation,
            get_logger,
        )
        from openhands.sdk.conversation.response_utils import get_agent_final_response
        from openhands.sdk.event import ActionEvent
        from openhands.sdk.event.llm_convertible import MessageEvent
        from openhands.sdk.event.llm_convertible.observation import ObservationEvent
        from openhands.tools.preset.default import get_default_tools
    except ImportError as e:
        return {
            "code_answer": (
                f"openhands-sdk import failed: {e}. "
                "Install with `uv pip install openhands-sdk openhands-tools` "
                "or switch OPENHANDS_BACKEND to 'stub' / 'cloud'."
            ),
            "openhands_status": "error",
            "openhands_backend": "sdk",
        }

    # ----- Config -------------------------------------------------------
    chat_model = os.environ.get("MODEL_NAME", "qwen2.5:7b")
    sdk_model = os.environ.get("OPENHANDS_MODEL_NAME") or f"ollama_chat/{chat_model}"

    # base_url: explicit OPENHANDS_BASE_URL wins. Otherwise, only auto-derive
    # from OLLAMA_BASE_URL if we're actually using an ollama_chat model.
    # For openrouter/openai/anthropic prefixes, litellm handles the URL itself.
    sdk_base = os.environ.get("OPENHANDS_BASE_URL")
    if not sdk_base and sdk_model.startswith("ollama_chat/"):
        chat_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        sdk_base = chat_base.rstrip("/").removesuffix("/v1")

    # api_key: OPENHANDS_API_KEY wins. Otherwise litellm picks up
    # provider-specific env vars (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, etc.).
    sdk_api_key = os.environ.get("OPENHANDS_API_KEY")

    max_iter = int(os.environ.get("OPENHANDS_MAX_ITERATIONS", "20"))
    disable_browser = os.environ.get("OPENHANDS_DISABLE_BROWSER", "1") == "1"
    workspace_root = Path(os.environ.get("OPENHANDS_WORKSPACE_ROOT", "workspace"))

    instance_id = state.get("swe_instance_id") or "ad_hoc"
    workspace_path = workspace_root / f"openhands_{instance_id}"
    workspace_path.mkdir(parents=True, exist_ok=True)

    user_prompt = state.get("code_question") or state.get("main_task") or ""
    if not user_prompt:
        return {
            "code_answer": "No task provided",
            "openhands_status": "error",
            "openhands_backend": "sdk",
        }

    print(f"[OPENHANDS-SDK] model={sdk_model}, base_url={sdk_base or '(provider default)'}")
    print(f"[OPENHANDS-SDK] api_key={'set' if sdk_api_key else '(env-derived)'}")
    print(f"[OPENHANDS-SDK] workspace={workspace_path}, max_iter={max_iter}")
    print(f"[OPENHANDS-SDK] instance_id={instance_id}")

    # ----- Build LLM, Agent, trace callback -----------------------------
    try:
        llm_kwargs = dict(
            model=sdk_model,
            usage_id=f"condition-c-{instance_id}",
            temperature=0.0,
            num_retries=2,
            timeout=600,
            # Disable reasoning/thinking-mode parameters - the SDK defaults
            # target Claude/GPT-4 with extended thinking, which Ollama 7B
            # models reject with: '{model} does not support thinking'.
            reasoning_effort=None,
            extended_thinking_budget=0,
            enable_encrypted_reasoning=False,
        )
        if sdk_base:
            llm_kwargs["base_url"] = sdk_base
            # ollama_base_url only matters for the ollama_chat provider
            if sdk_model.startswith("ollama_chat/"):
                llm_kwargs["ollama_base_url"] = sdk_base
        if sdk_api_key:
            llm_kwargs["api_key"] = sdk_api_key

        llm = LLM(**llm_kwargs)

        tools = get_default_tools(enable_browser=not disable_browser)
        agent = Agent(llm=llm, tools=tools)

        # Trace buffer captured by the per-event callback below. Each entry is
        # a serializable dict so it can be passed through ChatState (project
        # rule: state values must be string-serializable).
        action_trace: list[Dict[str, Any]] = []
        step_counter = {"i": 0}

        def _trace_cb(event):
            """Per-event callback: capture actions and their observations."""
            try:
                if isinstance(event, ActionEvent) and getattr(event, "source", "") == "agent":
                    step_counter["i"] += 1
                    action_obj = getattr(event, "action", None)
                    summary = (
                        action_obj.model_dump_json()[:1500]
                        if action_obj is not None and hasattr(action_obj, "model_dump_json")
                        else str(action_obj)[:1500]
                    )
                    action_trace.append({
                        "step": step_counter["i"],
                        "type": "action",
                        "tool_name": getattr(event, "tool_name", "unknown"),
                        "action_summary": summary,
                    })
                elif isinstance(event, ObservationEvent):
                    obs = getattr(event, "observation", None)
                    obs_text = (
                        obs.model_dump_json()[:1500]
                        if obs is not None and hasattr(obs, "model_dump_json")
                        else str(obs)[:1500]
                    )
                    action_trace.append({
                        "step": step_counter["i"],
                        "type": "observation",
                        "tool_name": getattr(event, "tool_name", "unknown"),
                        "observation_summary": obs_text,
                    })
            except Exception as cb_err:
                # Never let a callback bug kill the run - just note it.
                print(f"[OPENHANDS-SDK] trace callback error: {cb_err}")

        conversation = Conversation(
            agent=agent,
            workspace=workspace_path,
            callbacks=[_trace_cb],
            max_iteration_per_run=max_iter,
            stuck_detection=True,
        )

        # ----- Run ------------------------------------------------------
        t0 = time.time()
        conversation.send_message(user_prompt)
        conversation.run()
        elapsed = time.time() - t0
        print(f"[OPENHANDS-SDK] run finished in {elapsed:.1f}s, captured {len(action_trace)} trace entries")

        # ----- Extract final answer -------------------------------------
        try:
            events = list(conversation.state.events)
        except Exception:
            events = []
        final_text = get_agent_final_response(events)
        if not final_text:
            final_text = (
                "OpenHands agent ended without a final message. "
                f"Captured {len(action_trace)} actions; check trace for partial work."
            )

        return {
            "code_answer": final_text,
            "openhands_status": "success",
            "openhands_backend": "sdk",
            "openhands_action_trace": action_trace,
            "openhands_wall_time_s": round(elapsed, 2),
            "openhands_step_count": step_counter["i"],
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[OPENHANDS-SDK] run raised: {e}\n{tb}")
        return {
            "code_answer": f"OpenHands SDK error: {type(e).__name__}: {e}",
            "openhands_status": "error",
            "openhands_backend": "sdk",
        }


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def create_openhands_chain(timeout: int = 120) -> Callable[[Dict], Dict]:
    """
    Factory called by graph.py:build_graph(variant='c').

    Reads OPENHANDS_BACKEND once at chain-creation time (graph build), so the
    backend is fixed for the lifetime of the compiled graph. To switch
    backends, set the env var BEFORE calling build_graph.

    Returns:
        A callable taking ChatState dict and returning a dict that always
        contains "code_answer" plus diagnostic keys ("openhands_status",
        "openhands_backend").
    """
    backend = _resolve_backend()
    print(f"[OPENHANDS] create_openhands_chain() bound to backend='{backend}'")

    def openhands_invoke(state: Dict) -> Dict:
        if backend == "stub":
            return _invoke_stub(state)
        if backend == "cloud":
            return _invoke_cloud(state, timeout=timeout)
        if backend == "sdk":
            return _invoke_sdk(state)
        # Should never reach here because _resolve_backend normalizes input
        return {
            "code_answer": f"Unknown OpenHands backend: {backend}",
            "openhands_status": "error",
            "openhands_backend": backend,
        }

    return openhands_invoke


def create_openhands_sync(timeout: int = 120) -> Callable[[Dict], Dict]:
    """Backwards-compat alias preserved from the previous module."""
    return create_openhands_chain(timeout=timeout)
