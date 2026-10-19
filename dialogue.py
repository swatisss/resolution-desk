"""What was said earlier in this chat, kept in the desk's session-scoped `conversation` memory.

Pure functions, no I/O. Vera recalls the namespace before the agent runs (`current_run().memory`);
the desk decides what a turn is, how much of it to keep and how to use it (plan v3.32).
"""

from dataclasses import dataclass
from typing import Any

NAMESPACE = "conversation"
KEEP = 300  # characters kept of what was asked, said and found: a record stays small


@dataclass(frozen=True)
class Turn:
    asked: str
    replied: str
    status: str = ""
    order_id: str | None = None


def turn_record(
    asked: str, replied: str, status: str = "", order_id: str | None = None
) -> dict[str, Any]:
    """What the desk keeps of one turn."""
    record: dict[str, Any] = {"asked": asked[:KEEP], "replied": replied[:KEEP]}
    if status:
        record["status"] = status[:KEEP]
    if order_id:
        record["order_id"] = order_id
    return record


def turns_from(items: Any) -> list[Turn]:
    """The earlier turns, oldest first. A record that is not a turn is skipped."""
    mine = sorted((i for i in items if i.namespace == NAMESPACE), key=lambda i: -i.age_s)
    turns = []
    for item in mine:
        v = item.value
        if (
            isinstance(v, dict)
            and isinstance(v.get("asked"), str)
            and isinstance(v.get("replied"), str)
        ):
            order = v.get("order_id")
            turns.append(
                Turn(
                    v["asked"],
                    v["replied"],
                    str(v.get("status", "")),
                    str(order) if order else None,
                )
            )
    return turns


def last_order(turns: list[Turn]) -> str | None:
    """The order this chat was most recently about."""
    return next((t.order_id for t in reversed(turns) if t.order_id), None)


def turn_facts(turns: list[Turn]) -> str | None:
    """For the model, as data to use: not instructions."""
    if not turns:
        return None
    lines = ["Earlier in this chat:"]
    for t in turns:
        lines.append(f"- Customer: {t.asked}")
        if t.status:
            lines.append(f"  Order status found: {t.status}")
        lines.append(f"  Desk replied: {t.replied}")
    return "\n".join(lines)
