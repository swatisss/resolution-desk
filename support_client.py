"""Asking another agent for help, the way v3.10 describes: plain HTTP to the gateway.

No Vera SDK, no key. The request carries only the `traceparent` the gateway sent with the
run being served; the gateway works out who is calling from the pod's address and that
trace, checks the manifest's `calls:`, and runs the other agent as a child run with its own
checks, guardrails, metering and audit. A call held for a person's approval is polled.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

import httpx

log = logging.getLogger("resolution-desk")

Status = Literal["answered", "pending", "declined", "unavailable"]
WAITING = ("running", "suspended")


@dataclass(frozen=True)
class Reply:
    status: Status
    text: str
    run_id: str | None = None

    def line(self) -> str:
        """One line for the case note."""
        if self.status == "answered":
            return self.text
        if self.status == "pending":
            return f"waiting for a person to approve it (support-bot run {self.run_id})"
        if self.status == "declined":
            return f"declined: {self.text}"
        return f"not available right now ({self.text})"


class SupportDesk:
    """The support-bot agent, reached through the gateway."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        callee: str = "support-bot",
        poll_every_s: float = 1.0,
        patience_s: float = 300.0,
    ) -> None:
        self._http, self._callee = http, callee
        self._poll_every_s, self._patience_s = poll_every_s, patience_s

    async def ask(self, message: str, traceparent: str | None) -> Reply:
        try:
            return await self._ask(message, {"traceparent": traceparent or ""})
        except httpx.HTTPError:
            log.exception("calling %s through the gateway failed", self._callee)
            return Reply("unavailable", "the gateway could not be reached")

    async def _ask(self, message: str, headers: dict[str, str]) -> Reply:
        resp = await self._http.post(
            f"/v1/agents/{self._callee}/call", json={"input": message}, headers=headers
        )
        waited = 0.0
        while resp.status_code < 400 and resp.json().get("status") in WAITING:
            if waited >= self._patience_s:
                return Reply("pending", "", resp.json().get("run_id"))
            await asyncio.sleep(self._poll_every_s)  # running, or held for approval
            waited += self._poll_every_s
            run_id = resp.json()["run_id"]
            resp = await self._http.get(f"/v1/agent-calls/{run_id}", headers=headers)
        body = resp.json()
        if resp.status_code >= 400:
            detail = body.get("detail") or body.get("reason") or f"status {resp.status_code}"
            return Reply("unavailable", str(detail))
        if body.get("status") == "blocked":
            return Reply("declined", str(body.get("reason") or "no reason given"))
        return Reply("answered", str(body.get("output") or ""), body.get("run_id"))
