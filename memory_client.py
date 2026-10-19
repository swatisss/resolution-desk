"""Writing to the desk's memory: plain HTTP to the gateway, the way v3.24 describes.

No Vera SDK, no key. The request carries only the `traceparent` the gateway sent with the run
being served; the gateway works out who is calling and, from the run, whose records these are.
There is no field to name another customer. Memory is a convenience for the next case, so a
failure is logged and never fails this one.
"""

import logging
from typing import Any

import httpx

log = logging.getLogger("resolution-desk")


class CaseHistory:
    def __init__(self, http: httpx.AsyncClient, namespace: str = "case-history") -> None:
        self._http, self._namespace = http, namespace

    async def remember(
        self, record: dict[str, Any], tags: list[str], call_id: str, traceparent: str | None
    ) -> bool:
        """Append one case; the same `call_id` writes it once, even if the node runs again."""
        try:
            resp = await self._http.post(
                f"/v1/memory/{self._namespace}/records",
                json={"value": record, "tags": tags, "call_id": call_id},
                headers={"traceparent": traceparent or ""},
            )
        except httpx.HTTPError:
            log.exception("could not reach the gateway to remember the case")
            return False
        if resp.status_code >= 400:
            log.warning("memory refused the case: %s %s", resp.status_code, resp.text[:200])
            return False
        return True
