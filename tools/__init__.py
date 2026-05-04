from .calculator import calculator
from .web_search import web_search
from .python_executor import python_executor
from .ocr_reader import ocr_reader
from .registry import ToolRegistry


def get_tool_registry(dataset_name: str = None):
    """Factory: returns the right tool registry for a given dataset name.

    For tau2_<domain> datasets, returns a domain-scoped Tau2ToolRegistry that
    exposes the domain's task-relevant tools (airline / retail / telecom each
    ship their own toolkit). For all other datasets returns the default 4-tool
    ToolRegistry.
    """
    if dataset_name and dataset_name.startswith("tau2_"):
        from tau2_code.src.tool_registry import Tau2ToolRegistry
        return Tau2ToolRegistry(domain=dataset_name.removeprefix("tau2_"))
    return ToolRegistry()


__all__ = ['calculator', 'web_search', 'python_executor', 'ocr_reader', 'ToolRegistry', 'get_tool_registry']
