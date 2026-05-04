"""ConfiguredAgent - wraps a fully-configured workflow into a one-method dialog agent.

Idea: the structure policy has already chosen (workflow, tools, budgets); the prompt
policy has already chosen the per-role atoms. We bind all of those into a callable
that takes a conversation history and returns the agent's next action string.

The action string returned by .respond() is whatever our final answerer emits — either
a plain message to the user or a tool-call line in our standard format
(TOOL: name || QUERY: {...}). The rollout layer is responsible for translating
that into tau2's expected gym-action format.
"""
from typing import Dict, List, Optional


_PER_TURN_SYSTEM_INSTRUCTION = (
    "You are an interactive customer-support agent. The conversation so far is shown "
    "as the input. Decide ONE next action: either send a single message to the user, "
    "or call exactly one tool. Do not chain multiple actions in one response.\n\n"
    "If you choose to call a tool, output ONLY a single line in this exact format:\n"
    "  TOOL: <tool_name> || QUERY: <JSON object of arguments>\n"
    "Do not add commentary before or after the tool line. Tool arguments must be a JSON "
    "object whose keys exactly match the tool's parameter names. JSON must be a single "
    "line (no internal newlines).\n\n"
    "If you choose to send a message instead, output the message text only — no "
    "TOOL: prefix, no quotes around the message."
)


def _build_tool_descriptions_block(
    descriptions: dict,
    names: Optional[List[str]] = None,
    *,
    tool_registry=None,
) -> str:
    """Render the selected tool descriptions for the system prompt.

    If `tool_registry` exposes semantic groups (Tau2ToolRegistry), tools are
    rendered organized by group with each group's capability description, so
    the LLM understands what each group is for and which tools belong to it.
    Only groups that contain at least one selected tool are rendered.

    If `tool_registry` is None or has no groups, falls back to a flat list.
    Returns "" if the resulting selection is empty.
    """
    if not descriptions:
        return ""
    selected = list(descriptions.keys()) if names is None else [n for n in names if n in descriptions]
    if not selected:
        return ""

    # Group-aware rendering when the registry exposes groups.
    if tool_registry is not None and hasattr(tool_registry, "list_groups"):
        active_groups = tool_registry.groups_active_in_selection(selected)
        if active_groups:
            sel_set = set(selected)
            sections = []
            for gname in active_groups:
                gdesc = tool_registry.group_description(gname)
                gtools = [t for t in tool_registry.tools_in_group(gname) if t in sel_set]
                if not gtools:
                    continue
                heading = f"### Group: {gname}"
                if gdesc:
                    heading += f" — {gdesc}"
                body = "\n\n".join(descriptions[t] for t in gtools)
                sections.append(f"{heading}\n{body}")
            return (
                "## Available Tools (organized by capability group)\n"
                "Use any tool from a listed group. Call at most one per turn.\n\n"
                + "\n\n".join(sections)
            )

    # Flat fallback (default datasets or empty groups).
    items = [descriptions[n] for n in selected]
    return (
        "## Available Tools (call exactly one per turn, or send a plain message instead)\n"
        + "\n\n".join(items)
    )


class ConfiguredAgent:
    """Bind a workflow + atoms + tau2 tool registry into .respond(history)."""

    def __init__(
        self,
        workflow,
        worker,
        tool_registry,
        prompt_suffixes: Dict[str, str],
        agent1_tools: List[str],
        agent2_tools: List[str],
        agent1_tokens: int,
        agent2_tokens: int,
        answerer_tokens: int,
        agent1_budget: int = 1,
        agent2_budget: int = 1,
        answerer_budget: int = 1,
    ):
        self.workflow = workflow
        self.worker = worker
        self.tool_registry = tool_registry

        self.prompt_suffixes = dict(prompt_suffixes or {})

        # Preserve the structure policy's per-agent tool selection (subset of the
        # domain toolkit). We still pass EMPTY lists to the workflow so the worker's
        # built-in tool-prompt scaffolding is skipped and the workflow's
        # _process_tool_calls short-circuits — tau2's gym env handles execution.
        # Tool descriptions for the SELECTED subset are injected via prompt_suffix
        # below so the LLM only sees the tools the policy chose for each role.
        self._selected_agent1_tools: List[str] = list(agent1_tools or [])
        self._selected_agent2_tools: List[str] = list(agent2_tools or [])
        self.agent1_tools: list = []
        self.agent2_tools: list = []
        self.agent1_tokens = int(agent1_tokens)
        self.agent2_tokens = int(agent2_tokens)
        self.answerer_tokens = int(answerer_tokens)
        self.agent1_budget = int(agent1_budget)
        self.agent2_budget = int(agent2_budget)
        self.answerer_budget = int(answerer_budget)

        # Per-role tool description blocks. Reasoner sees agent1's selection,
        # verifier sees agent2's, answerer sees the union since it's what
        # actually emits the action sent to env.step().
        full_descs = (
            self.tool_registry.get_descriptions_dict()
            if hasattr(self.tool_registry, "get_descriptions_dict") else {}
        )
        union = list(dict.fromkeys(self._selected_agent1_tools + self._selected_agent2_tools))
        per_role_blocks = {
            "reasoner": _build_tool_descriptions_block(
                full_descs, self._selected_agent1_tools, tool_registry=self.tool_registry),
            "verifier": _build_tool_descriptions_block(
                full_descs, self._selected_agent2_tools, tool_registry=self.tool_registry),
            "answerer": _build_tool_descriptions_block(
                full_descs, union, tool_registry=self.tool_registry),
        }

        for role, tool_block in per_role_blocks.items():
            addendum = "\n\n".join(s for s in (tool_block, _PER_TURN_SYSTEM_INSTRUCTION) if s)
            existing = self.prompt_suffixes.get(role, "") or ""
            self.prompt_suffixes[role] = (
                (existing + "\n\n" if existing else "") + addendum
            )

        self.cumulative_tokens = 0
        self.cumulative_steps = 0
        self.cumulative_tool_calls = 0

    def respond(self, conversation_history: str) -> str:
        """Run one full workflow execution treating `conversation_history` as the question.

        Returns the agent's next action string (plain message or tool-call line).
        """
        final_text, exec_info = self.workflow.execute(
            question=conversation_history,
            agent1_tools=self.agent1_tools,
            agent1_budget=self.agent1_budget,
            agent2_tools=self.agent2_tools,
            agent2_budget=self.agent2_budget,
            answerer_budget=self.answerer_budget,
            agent1_tokens=self.agent1_tokens,
            agent2_tokens=self.agent2_tokens,
            answerer_tokens=self.answerer_tokens,
            prompt_suffixes=self.prompt_suffixes,
        )
        self.cumulative_steps += int(exec_info.get("steps", 1))
        self.cumulative_tokens += int(exec_info.get("total_tokens", 0))
        self.cumulative_tool_calls += int(exec_info.get("tools_count", 0))
        return (final_text or "").strip()

    def stats(self) -> Dict[str, int]:
        return {
            "steps": self.cumulative_steps,
            "total_tokens": self.cumulative_tokens,
            "tools_count": self.cumulative_tool_calls,
        }
