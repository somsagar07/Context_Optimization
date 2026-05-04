"""Tau2ToolRegistry - ToolRegistry-shaped facade over a tau2 domain's toolkit.

The agent's tool calls are NOT executed by us in tau2 mode. Tau2's gym env parses and
executes them itself when we pass the agent's action string to env.step(). This
registry's role is therefore:

  1) Expose the domain's tool *names* so existing workflow code can pass them to
     `worker.reason(tools=[...])` — this triggers the tool-prompt scaffolding inside
     the worker.
  2) Provide rich tool descriptions (`additional_tool_descriptions` dict) the worker
     can splice into the LLM prompt so the LLM knows what each tool does and how
     to invoke it.
  3) `parse_and_execute()` is a NO-OP in tau2 mode: tool execution belongs to
     env.step(). The agent's raw output (which may contain a tool call) is what the
     rollout passes to the env.
"""
from typing import List, Dict


def _tau2_tool_block(tool) -> str:
    """Render one tau2 Tool object into the prompt-ready format our worker expects.

    Worker code looks up `TOOL_DESCRIPTIONS[name]`; we provide the full block keyed
    by tool name so it splices in cleanly.
    """
    sig = ""
    try:
        sig = tool.to_str().strip()
    except Exception:
        sig = ""
    short = (getattr(tool, "short_desc", "") or "").strip()
    long_desc = (getattr(tool, "long_desc", "") or "").strip()
    pieces = [f"- {tool.name}: {short or 'Domain tool.'}"]
    if long_desc:
        pieces.append(f"  Details: {long_desc}")
    if sig:
        pieces.append(f"  Signature: {sig}")
    pieces.append(
        f"  Format: TOOL: {tool.name} || QUERY: <JSON object of arguments>\n"
        f"  Example: TOOL: {tool.name} || QUERY: {{\"arg1\": \"value1\"}}"
    )
    return "\n".join(pieces)


class Tau2ToolRegistry:
    """Domain-scoped tool registry. Same interface as our default ToolRegistry."""

    def __init__(self, domain: str):
        from tau2.registry import registry as tau2_registry
        from .tool_groups import get_groups_for_domain

        self.domain = domain
        self._env = tau2_registry.get_env_constructor(domain)()
        self._tools = list(self._env.get_tools())
        self._tools_by_name = {t.name: t for t in self._tools}

        # ToolRegistry-shape: a dict mapping tool name -> callable. We don't actually
        # call these from our side (gym does), but workflow code may probe for membership.
        self.available_tools = {t.name: self._make_noop_callable(t.name) for t in self._tools}

        # Pre-rendered descriptions for worker.additional_tool_descriptions
        self.descriptions: Dict[str, str] = {
            t.name: _tau2_tool_block(t) for t in self._tools
        }

        # Semantic groups: ordered dict of group_name -> {description, tools}.
        # Bit i of the structure policy's tool action selects group i.
        self._groups = get_groups_for_domain(domain, list(self.available_tools.keys()))
        self._group_names: List[str] = list(self._groups.keys())

    @staticmethod
    def _make_noop_callable(name: str):
        def _noop(*_args, **_kwargs):
            return f"[Tau2ToolRegistry] Tool {name!r} is executed by the gym env, not by us."
        return _noop

    # ---- ToolRegistry interface ----
    def list_tools(self) -> List[str]:
        return list(self.available_tools.keys())

    def execute(self, tool_name: str, query: str) -> str:
        # Not used in tau2 mode — kept for interface compatibility.
        return self._make_noop_callable(tool_name)(query)

    def parse_and_execute(self, response_text: str, allowed_tools: list) -> str:
        # Critical no-op: tau2's env.step(action) parses tool calls, not us.
        # Returning empty string means workflow._process_tool_calls leaves the agent's
        # output untouched, which is exactly what we want to forward to env.step().
        return ""

    # ---- Tau2 extras ----
    def get_tools_objects(self):
        return list(self._tools)

    def get_descriptions_dict(self) -> Dict[str, str]:
        return dict(self.descriptions)

    # ---- Group interface (used by the structure policy + prompt injection) ----
    def list_groups(self) -> List[str]:
        return list(self._group_names)

    def num_groups(self) -> int:
        return len(self._group_names)

    def tools_in_group(self, group_name: str) -> List[str]:
        spec = self._groups.get(group_name)
        return list(spec["tools"]) if spec else []

    def group_description(self, group_name: str) -> str:
        spec = self._groups.get(group_name)
        return str(spec["description"]) if spec else ""

    def decode_group_mask(self, idx: int) -> List[str]:
        """Decode a bitmask over groups into a flat ordered list of tool names.

        Bit i selects group i (in self._group_names order). Tools are emitted in
        registry order, deduped, so the same idx always yields the same list.
        """
        selected: List[str] = []
        seen: set = set()
        for i, gname in enumerate(self._group_names):
            if not (idx & (1 << i)):
                continue
            for t in self._groups[gname]["tools"]:
                if t in seen:
                    continue
                seen.add(t)
                selected.append(t)
        return selected

    def groups_active_in_selection(self, tool_names: List[str]) -> List[str]:
        """Return the ordered group names that contain any of the given tools.
        Used by the prompt-injection layer to render tools organized by group."""
        wanted = set(tool_names)
        out = []
        for gname in self._group_names:
            tools = self._groups[gname]["tools"]
            if any(t in wanted for t in tools):
                out.append(gname)
        return out
