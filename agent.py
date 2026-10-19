"""Customer Resolution Desk: an enterprise case agent that works with support-bot.

A service manager pastes in a customer's case. The desk finds the order, asks support-bot
(through the Vera gateway, agent to agent) for its status and, when the customer wants
money back, asks it for the refund, which waits for a person's approval. A model drafts
the reply to the customer. The result is one case note.

The agent holds no model key and no credential: model calls go through VeraChatModel,
and the call to support-bot is plain HTTP to the gateway (support_client.py).
"""

import os
from typing import Any

import httpx
from cases import (
    DRAFT_PROMPT,
    NEED_ORDER,
    Case,
    case_note,
    customer_reply,
    draft_facts,
    lookup_request,
    parse_case,
    refund_request,
)
from customer import Brief, brief_facts, brief_from, case_record, greeting, on_file, refund_flag
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from memory_client import CaseHistory
from support_client import Reply, SupportDesk

from vera_sdk import VeraChatModel, current_run


class DeskState(MessagesState):
    note: str
    flag: str | None
    brief: dict[str, Any] | None
    case: dict[str, Any]
    lookup: dict[str, Any]
    refund: dict[str, Any]


def _case(state: DeskState) -> Case:
    return Case(**state["case"])


def _brief(state: DeskState) -> Brief | None:
    raw = state.get("brief")
    return Brief(**raw) if raw else None


def _reply(raw: dict[str, Any]) -> Reply:
    return Reply(**raw)


def build_graph(gateway: httpx.AsyncClient | None = None, patience_s: float = 300.0) -> Any:
    """The compiled graph. `gateway` is a client whose base URL is the gateway's in-cluster
    address; left out, one is made from $VERA_GATEWAY_URL."""
    http = gateway or httpx.AsyncClient(
        base_url=os.environ["VERA_GATEWAY_URL"], timeout=httpx.Timeout(15.0)
    )
    support = SupportDesk(http, patience_s=patience_s)
    history = CaseHistory(http)
    model = VeraChatModel(model="claude-haiku-5-5")

    async def recall(state: DeskState) -> dict[str, Any]:
        """What Vera recalled for this customer before the agent ran (nothing for a stranger)."""
        brief = brief_from(current_run().memory)
        return {"brief": brief.__dict__ if brief else None}

    async def intake(state: DeskState) -> dict[str, Any]:
        case = parse_case(state["messages"][-1].text)
        update: dict[str, Any] = {"case": case.__dict__}
        if case.order_id is None:
            update["messages"] = [AIMessage(greeting(_brief(state)) + NEED_ORDER)]
        return update

    async def check_order(state: DeskState) -> dict[str, Any]:
        reply = await support.ask(lookup_request(_case(state)), current_run().traceparent)
        return {"lookup": reply.__dict__}

    async def request_refund(state: DeskState) -> dict[str, Any]:
        reply = await support.ask(refund_request(_case(state)), current_run().traceparent)
        return {"refund": reply.__dict__}

    async def compose(state: DeskState) -> dict[str, Any]:
        case = _case(state)
        lookup = _reply(state["lookup"]).line()
        refund = _reply(state["refund"]).line() if state.get("refund") else None
        brief = _brief(state)
        facts = draft_facts(case, lookup, refund, brief_facts(brief) if brief else None)
        draft = await model.ainvoke([SystemMessage(DRAFT_PROMPT), HumanMessage(facts)])
        flag = refund_flag(brief, case.intent)
        # The customer is told only the reply. The case note, with what the desk knows and any
        # flag, goes to its memory, where the person handling the case reads it.
        note = case_note(
            case, lookup, refund, "", on_file=on_file(brief) if brief else None, flag=flag
        )
        reply = customer_reply(greeting(brief), draft.text)
        return {"messages": [AIMessage(reply)], "note": note, "flag": flag}

    async def remember(state: DeskState) -> dict[str, Any]:
        """Keep the case for next time. Without a customer there is nobody to keep it for."""
        ctx = current_run()
        if ctx.run.end_user:
            case = _case(state)
            refund = _reply(state["refund"]).line() if state.get("refund") else None
            record = case_record(
                case.order_id or "",
                case.intent,
                refund,
                case.ref,
                state.get("note", ""),
                state.get("flag"),
            )
            await history.remember(record, [case.intent], ctx.run.run_id, ctx.traceparent)
        return {}

    def after_intake(state: DeskState) -> str:
        return "check_order" if state["case"]["order_id"] else END

    def after_lookup(state: DeskState) -> str:
        wants_refund = state["case"]["intent"] == "refund"
        found = state["lookup"]["status"] == "answered"
        return "request_refund" if wants_refund and found else "compose"

    graph = StateGraph(DeskState)
    graph.add_node("recall", recall)
    graph.add_node("intake", intake)
    graph.add_node("check_order", check_order)
    graph.add_node("request_refund", request_refund)
    graph.add_node("compose", compose)
    graph.add_node("remember", remember)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "intake")
    graph.add_conditional_edges("intake", after_intake, ["check_order", END])
    graph.add_conditional_edges("check_order", after_lookup, ["request_refund", "compose"])
    graph.add_edge("request_refund", "compose")
    graph.add_edge("compose", "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=InMemorySaver())
