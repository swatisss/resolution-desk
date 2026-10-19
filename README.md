# Customer Resolution Desk

An enterprise case agent for the Vera demo. A service manager pastes in a customer's case;
the desk finds the order, asks **support-bot** (agent to agent, through the gateway) for its
status and, when the customer wants money back, for the refund, which waits for a person's
approval. A model drafts the reply to the customer, and the desk answers with that reply only (greeting
the customer by name when it remembers them). The case note, with what it knows and any flag for a
person, is kept in its `case-history` memory, where the Playground's side panel shows it.

| File | What it is |
|---|---|
| `vera.yaml` | the manifest: models, `calls: [support-bot]`, the agent's own guardrails, golden cases |
| `agent.py` | the LangGraph graph: recall → intake → check order → (request refund) → compose → remember |
| `cases.py` | reading a case, the reply to the customer and the case note (pure functions) |
| `customer.py` | what the desk knows about a customer, and what it does with it (pure functions) |
| `memory_client.py` | writing the case to the customer's memory: plain HTTP to the gateway, no key |
| `seed/` | two synthetic customers (Priya Nair, Marc Dubois) that Vera loads into memory for the demo |
| `support_client.py` | the call to support-bot: plain HTTP to the gateway, no SDK, no key |

Zip it and submit it as code: `cd examples && zip -r resolution-desk.zip resolution-desk`
(PowerShell: `Compress-Archive -Path examples\resolution-desk\* -DestinationPath resolution-desk.zip`).
The step-by-step demo is [docs/demo-resolution-desk.md](../../docs/demo-resolution-desk.md).
