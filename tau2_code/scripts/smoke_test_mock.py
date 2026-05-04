"""End-to-end smoke test for the tau2 integration.

Runs ONE mock-domain task with the OpenRouterDirectWorkflow, no atoms, full dialog mode.
Verifies that:
  - Tau2Dataset loads tasks
  - Tau2ToolRegistry exposes tools
  - ConfiguredAgent + workflow.execute() produces an action per turn
  - tau2_dialog_rollout runs the loop, captures reward, returns transcript

Run from the repo root:
    python tau2_code/scripts/smoke_test_mock.py
"""
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO, ".env"))

from agents_system.worker import OpenRouterWorker
from agents_system.workflows import get_openrouter_workflow

from tau2_code.src import Tau2Dataset, Tau2ToolRegistry, ConfiguredAgent, tau2_dialog_rollout


def main():
    print("=== Tau2 smoke test (mock domain) ===")

    ds = Tau2Dataset(domain="mock", split="base")
    print(f"  loaded {len(ds)} mock tasks")
    question, task_id = ds.get_sample()
    print(f"  sampled task_id={task_id}")
    print(f"  task description (first 300):\n  {question[:300]!r}\n")

    tools = Tau2ToolRegistry(domain="mock")
    print(f"  domain tools ({len(tools.list_tools())}): {tools.list_tools()}\n")

    # NOTE: OpenRouterWorker (our raw HTTP client) wants bare model IDs ('openai/gpt-4o-mini').
    # The 'openrouter/' prefix is only for tau2's LiteLLM-based user simulator,
    # configured via OPENROUTER_USER_MODEL env var.
    worker = OpenRouterWorker(model_name="openai/gpt-4o-mini")

    workflow = get_openrouter_workflow(workflow_depth=0, worker=worker, tools_registry=tools)

    agent = ConfiguredAgent(
        workflow=workflow,
        worker=worker,
        tool_registry=tools,
        prompt_suffixes={},                          # no atoms for the smoke test
        agent1_tools=tools.list_tools(),             # all domain tools available
        agent2_tools=tools.list_tools(),
        agent1_tokens=512,
        agent2_tokens=512,
        answerer_tokens=512,
    )

    transcript, exec_info, reward = tau2_dialog_rollout(
        agent,
        domain="mock",
        task_id=task_id,
        max_turns=8,
        solo_mode=False,
    )

    print(f"=== finished ===")
    print(f"  reward: {reward}")
    print(f"  cumulative tokens: {exec_info['total_tokens']}")
    print(f"  cumulative steps: {exec_info['steps']}")
    print(f"  cumulative tool_calls (parsed by us, not gym): {exec_info['tools_count']}")
    print(f"\n--- transcript (last 1200 chars) ---")
    print(transcript[-1200:])


if __name__ == "__main__":
    main()
