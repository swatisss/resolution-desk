"""Writing to the desk's memory, the case history and the conversation: plain HTTP to the gateway.

No Vera SDK, no key. The request carries only the `traceparent` the gateway sent with the run
being served; the gateway works out who is calling and, from the run, whose records these are.
There is no field to name another customer. Memory is a convenience for the next case, so a
failure is logged and never fails this one.
"""

import logging
from typing import Any

import httpx

log = logging.getLogger("resolution-desk")


class MemoryWriter:
    """Appends records to one of the desk's own namespaces."""

    def __init__(self, http: httpx.AsyncClient, namespace: str) -> None:
        self._http, self._namespace = http, namespace

    async def remember(
        self, record: dict[str, Any], tags: list[str], call_id: str, traceparent: str | None
    ) -> bool:
        """Append one record; one `call_id` writes once per run, even if the node reruns."""
        try:
            resp = await self._http.post(
                f"/v1/memory/{self._namespace}/records",
                json={"value": record, "tags": tags, "call_id": call_id},
                headers={"traceparent": traceparent or ""},
            )
        except httpx.HTTPError:
            log.exception("could not reach the gateway to remember a record in %s", self._namespace)
            return False
        if resp.status_code >= 400:
            log.warning(
                "memory refused a record in %s: %s %s",
                self._namespace,
                resp.status_code,
                resp.text[:200],
            )
            return False
        return True
