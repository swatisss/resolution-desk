"""The case a service manager writes up, and the notes Resolution Desk sends back.

Pure functions, no I/O: what an order number looks like, whether the customer wants a
status or a refund, the words sent to support-bot, and the case note that comes back.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

Intent = Literal["status", "refund"]

_ORDER = re.compile(r"(?:order|#)\s*(?:no\.?|number)?\s*#?\s*(\d{1,10})", re.IGNORECASE)
_CUSTOMER = re.compile(r"(?i:customer|client)\s+([A-Z][\w'-]+(?:\s[A-Z][\w'-]+)?)")
_REFUND_WORDS = (
    "refund", "money back", "damaged", "broken", "defective", "wrong item",
    "not as described", "never arrived", "return",
)  # fmt: skip

NEED_ORDER = (
    "I need an order number to look this up. Please add it to the case notes, "
    "for example “order 42”."
)


@dataclass(frozen=True)
class Case:
    ref: str
    order_id: str | None
    intent: Intent
    customer: str
    notes: str


def parse_case(notes: str) -> Case:
    order = _ORDER.search(notes)
    who = _CUSTOMER.search(notes)
    lowered = notes.lower()
    intent: Intent = "refund" if any(w in lowered for w in _REFUND_WORDS) else "status"
    digest = hashlib.sha1(notes.encode("utf-8"), usedforsecurity=False).hexdigest()
    return Case(
        ref=f"CR-{int(digest[:8], 16) % 9000 + 1000}",
        order_id=order.group(1) if order else None,
        intent=intent,
        customer=who.group(1) if who else "the customer",
        notes=notes.strip(),
    )


def lookup_request(case: Case) -> str:
    """What Resolution Desk asks support-bot first. The order number comes first."""
    return (
        f"Please lookup order {case.order_id} and tell me its current status and delivery "
        f"details. {case.customer} is asking for an update (case {case.ref})."
    )


def refund_request(case: Case) -> str:
    return (
        f"Please refund order {case.order_id} in full. {case.customer} reported a problem with "
        f"it and the service desk has reviewed case {case.ref}."
    )


DRAFT_PROMPT = (
    "You are a customer-resolution coordinator at Acme. Write a short, courteous reply to the "
    "customer, three sentences at most. Use only the facts you are given. Do not promise "
    "anything that is not stated, and do not mention internal systems."
)


def draft_facts(case: Case, lookup: str, refund: str | None) -> str:
    lines = [f"Case notes: {case.notes}", f"Order status: {lookup}"]
    if refund:
        lines.append(f"Refund: {refund}")
    return "\n".join(lines)


def case_note(case: Case, lookup: str, refund: str | None, draft: str) -> str:
    """The reply the service manager reads."""
    wanted = "refund" if case.intent == "refund" else "status update"
    parts = [
        f"Case {case.ref} · Customer Resolution Desk",
        f"Order {case.order_id} · {case.customer} · requested: {wanted}",
        "",
        f"Order status (support-bot): {lookup}",
    ]
    if refund:
        parts.append(f"Refund (support-bot): {refund}")
    parts += ["", "Suggested reply to the customer:", draft]
    return "\n".join(parts)
