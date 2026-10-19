"""What Resolution Desk knows about a customer, and what it does with it.

Pure functions, no I/O. Vera recalls the customer's records before the agent runs
(`current_run().memory`) and keeps them for a time to live; what to make of them is this
agent's business: the greeting, the contact preference, and when a pattern deserves a person.
"""

from dataclasses import dataclass, field
from typing import Any

# Business rule, the desk's own: a customer with this many refunds already on file (the case
# history lives 30 days) gets the next one flagged for a person to look at.
REFUND_REVIEW_AT = 2


@dataclass(frozen=True)
class Brief:
    name: str
    tier: str
    contact: str
    tone: str
    earlier_cases: list[str] = field(default_factory=list)
    earlier_refunds: int = 0

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else "there"


def _value(item: Any) -> dict[str, Any]:
    value = getattr(item, "value", None)
    return value if isinstance(value, dict) else {}


def brief_from(items: Any) -> Brief | None:
    """A brief from recalled memory, or None when nothing is known about this customer."""
    profile = next((_value(i) for i in items if i.namespace == "customer-profile"), {})
    cases = [_value(i) for i in items if i.namespace == "case-history"]
    if not profile and not cases:
        return None
    return Brief(
        name=str(profile.get("name", "")),
        tier=str(profile.get("tier", "")),
        contact=str(profile.get("preferred_contact", "")),
        tone=str(profile.get("tone", "")),
        earlier_cases=[str(c["summary"]) for c in cases if c.get("summary")],
        earlier_refunds=sum(1 for c in cases if c.get("intent") == "refund"),
    )


def greeting(brief: Brief | None) -> str:
    return f"Hi {brief.first_name}, " if brief and brief.name else ""


def brief_facts(brief: Brief) -> str:
    """For the model, as data to use: not instructions."""
    lines = [f"Customer: {brief.name or 'unknown'} ({brief.tier or 'no tier'} tier)"]
    if brief.contact:
        lines.append(f"Prefers to be contacted by: {brief.contact}")
    if brief.tone:
        lines.append(f"Tone they like: {brief.tone}")
    lines += [f"Earlier case: {c}" for c in brief.earlier_cases[:3]]
    return "\n".join(lines)


def refund_flag(brief: Brief | None, intent: str) -> str | None:
    """A refund for a customer who already has several on file is flagged, not refused."""
    if brief is None or intent != "refund" or brief.earlier_refunds < REFUND_REVIEW_AT:
        return None
    n = brief.earlier_refunds + 1
    return f"This would be refund number {n} in 30 days for this customer: review before approving."


def on_file(brief: Brief) -> str:
    bits = [
        brief.name or "customer",
        f"{brief.tier} tier" if brief.tier else "",
        f"prefers {brief.contact}" if brief.contact else "",
    ]
    return "Customer on file: " + " · ".join(b for b in bits if b)


def case_record(
    order_id: str,
    intent: str,
    refund: str | None,
    case_ref: str = "",
    note: str = "",
    flag: str | None = None,
) -> dict[str, Any]:
    """What the desk keeps about a case it worked: for the next time, and for the person who
    reads the case note beside the chat (Vera shows what an agent wrote to its memory)."""
    outcome = "refund requested" if refund else "answered"
    what = "refund" if intent == "refund" else "status question"
    record: dict[str, Any] = {
        "order_id": order_id,
        "intent": intent,
        "outcome": outcome,
        "summary": f"Order {order_id}: {what}; {outcome}.",
    }
    if case_ref:
        record["case_ref"] = case_ref
    if flag:
        record["flag"] = flag
    if note:
        record["note"] = note
    return record
