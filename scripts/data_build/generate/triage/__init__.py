"""Request triaging task: dependency-aware queue reasoning.

A scenario is a DAG of work requests. Each request has:
- a label/description
- zero or more dependencies (must complete before the request can start)
- a priority (urgent/normal/low)
- a status (pending/done/skipped/deprioritized)

Over a sequence of passages, requests arrive and status updates land
("Request A is now done", "Request D is being deprioritized"). The
model must reason over the current DAG state to answer:
- what_unblocked: which requests can be worked on now?
- what_blocks_X: which deps still block request X?
- is_X_ready: can request X be started?
- next_priority: what's the highest-priority unblocked request?
"""
