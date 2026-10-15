# Customer Resolution Desk

An enterprise case agent for the Vera demo. A service manager pastes in a customer's case;
the desk finds the order, asks **support-bot** (agent to agent, through the gateway) for its
status and, when the customer wants money back, for the refund, which waits for a person's
approval. A model drafts the reply to the customer; the result is one case note.

| File | What it is |
|---|---|
| `vera.yaml` | the manifest: models, `calls: [support-bot]`, the agent's own guardrails, golden cases |
| `agent.py` | the LangGraph graph: intake → check order → (request refund) → compose |
| `cases.py` | reading a case and writing the case note (pure functions) |
| `support_client.py` | the call to support-bot: plain HTTP to the gateway, no SDK, no key |

Zip it and submit it as code: `cd examples && zip -r resolution-desk.zip resolution-desk`
(PowerShell: `Compress-Archive -Path examples\resolution-desk\* -DestinationPath resolution-desk.zip`).
The step-by-step demo is [docs/demo-resolution-desk.md](../../docs/demo-resolution-desk.md).
