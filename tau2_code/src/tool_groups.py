"""Semantic tool groupings per tau2 domain.

The structure policy selects a *bitmask over groups* rather than over individual
tools. When a group is selected, ALL of its tools are made available to the
agent. This keeps the action space small and dense (2^G with G ~= 4) while still
letting the policy learn which capabilities the task needs.

Each group has:
  - description: a one-line capability summary, also injected into the prompt so
    the LLM understands what each group is for.
  - tools: ordered list of tau2 tool names that belong to this group.

We enforce at registry-construction time that every tool in the domain is
assigned to exactly one group; unassigned tools land in an auto-generated
"misc" group so we never silently drop one.
"""
from typing import Dict, List

# Each domain -> ordered dict of group_name -> {description, tools}.
# Order matters: bit i of the structure policy's tool action selects group i.
TOOL_GROUPS_BY_DOMAIN: Dict[str, Dict[str, Dict[str, object]]] = {
    "mock": {
        "user_lookup": {
            "description": "Look up users in the database.",
            "tools": ["get_users"],
        },
        "task_management": {
            "description": "Create and update user tasks.",
            "tools": ["create_task", "update_task_status"],
        },
        "escalation": {
            "description": "Hand the conversation off to a human agent.",
            "tools": ["transfer_to_human_agents"],
        },
    },
    "telecom": {
        "customer_lookup": {
            "description": "Identify the caller and pull their account details.",
            "tools": [
                "get_customer_by_phone",
                "get_customer_by_id",
                "get_customer_by_name",
                "get_details_by_id",
            ],
        },
        "line_management": {
            "description": "Suspend or resume a phone line on the account.",
            "tools": ["suspend_line", "resume_line"],
        },
        "billing": {
            "description": "Look up bills and request payment from the customer.",
            "tools": ["get_bills_for_customer", "send_payment_request"],
        },
        "data_services": {
            "description": "Inspect data usage and toggle roaming or data refuels.",
            "tools": [
                "get_data_usage",
                "enable_roaming",
                "disable_roaming",
                "refuel_data",
            ],
        },
        "escalation": {
            "description": "Hand the conversation off to a human agent.",
            "tools": ["transfer_to_human_agents"],
        },
    },
    "airline": {
        "search": {
            "description": "Search the catalog of airports and available flights.",
            "tools": [
                "list_all_airports",
                "search_direct_flight",
                "search_onestop_flight",
                "get_flight_status",
            ],
        },
        "account_lookup": {
            "description": "Identify the user and retrieve a reservation.",
            "tools": ["get_user_details", "get_reservation_details"],
        },
        "reservation_management": {
            "description": "Book, cancel, or modify reservations (flights, baggage, passengers).",
            "tools": [
                "book_reservation",
                "cancel_reservation",
                "update_reservation_flights",
                "update_reservation_baggages",
                "update_reservation_passengers",
            ],
        },
        "compensation": {
            "description": "Run numerical calculations and issue compensation certificates.",
            "tools": ["calculate", "send_certificate"],
        },
        "escalation": {
            "description": "Hand the conversation off to a human agent.",
            "tools": ["transfer_to_human_agents"],
        },
    },
    "retail": {
        "user_lookup": {
            "description": "Identify the shopper from name+zip or email and inspect their account.",
            "tools": [
                "find_user_id_by_name_zip",
                "find_user_id_by_email",
                "get_user_details",
                "modify_user_address",
            ],
        },
        "catalog": {
            "description": "Look up product, item, and category information.",
            "tools": [
                "list_all_product_types",
                "get_product_details",
                "get_item_details",
            ],
        },
        "order_inspection": {
            "description": "Read order status and details.",
            "tools": ["get_order_details"],
        },
        "order_modification": {
            "description": "Cancel, modify, exchange, or return orders. Also numeric calculation.",
            "tools": [
                "cancel_pending_order",
                "modify_pending_order_address",
                "modify_pending_order_items",
                "modify_pending_order_payment",
                "exchange_delivered_order_items",
                "return_delivered_order_items",
                "calculate",
            ],
        },
        "escalation": {
            "description": "Hand the conversation off to a human agent.",
            "tools": ["transfer_to_human_agents"],
        },
    },
}


def get_groups_for_domain(domain: str, all_tool_names: List[str]) -> Dict[str, Dict[str, object]]:
    """Return the ordered group dict for a domain, validated against the actual
    tool list. Tools not assigned to any group are collected into an auto "misc"
    group so we never silently drop them. Unknown tools (listed in TOOL_GROUPS
    but missing from the registry) are dropped with no error since the upstream
    toolkit may have changed.
    """
    raw = TOOL_GROUPS_BY_DOMAIN.get(domain)
    if raw is None:
        # Domain has no curated groups — fall back to one group containing
        # everything. Action size = 2^1 = 2 (use tools / don't).
        return {
            "all": {
                "description": f"All available tools for the {domain} domain.",
                "tools": list(all_tool_names),
            }
        }

    name_set = set(all_tool_names)
    out: Dict[str, Dict[str, object]] = {}
    assigned: set = set()
    for gname, spec in raw.items():
        kept = [t for t in spec["tools"] if t in name_set]
        if not kept:
            continue
        out[gname] = {"description": spec["description"], "tools": kept}
        assigned.update(kept)

    missing = [t for t in all_tool_names if t not in assigned]
    if missing:
        out["misc"] = {
            "description": "Other domain tools not covered by the groups above.",
            "tools": missing,
        }
    return out
