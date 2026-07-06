"""Triage task — scenario data model: DAG of requests with statuses."""

from __future__ import annotations

import string
from dataclasses import dataclass, field


# Pool of activity descriptions (the "what" of each request).
REQUEST_DESCRIPTIONS = (
    "set up the staging database",
    "migrate user accounts",
    "send notification emails",
    "audit access logs",
    "rotate API keys",
    "review the new design mockups",
    "draft the release notes",
    "verify the backup snapshots",
    "deploy to production",
    "investigate the latency spike",
    "schedule the maintenance window",
    "renew the SSL certificate",
    "publish the migration guide",
    "regenerate analytics dashboards",
    "interview the support hire",
    "prepare the budget forecast",
    "rebuild the staging environment",
    "merge the pending pull request",
    "tag the v3 release candidate",
    "respond to the procurement question",
    "consolidate vendor contracts",
    "patch the security advisory",
    "document the new endpoint",
    "test the rollback procedure",
)

# 26 single-letter labels (A..Z) — plenty for scenarios up to ~12 requests.
LABELS = tuple(string.ascii_uppercase)
PRIORITIES = ("urgent", "normal", "low")


@dataclass
class Request:
    label: str                    # "A", "B", ... — unique within scenario
    description: str
    deps: tuple[str, ...]         # labels of prerequisite requests
    priority: str
    status: str = "pending"       # pending | done | deprioritized


@dataclass
class StatusUpdate:
    """One status-change event in the scenario timeline."""
    label: str
    new_status: str               # "done" or "deprioritized"


@dataclass
class TriageScenario:
    scenario_idx: int
    requests: list[Request]                # initial set, in arrival order
    updates: list[StatusUpdate]            # ordered status updates after intake
    # Derived (computed in build_scenario after applying updates):
    final_status: dict[str, str] = field(default_factory=dict)
    # Per-passage evidence (which label's request was added/updated at each passage).
    passage_labels: list[str] = field(default_factory=list)


def _build_dag(rng, n: int) -> list[Request]:
    """Build a DAG of n requests. Each new request optionally depends on
    one earlier request (keeping the DAG simple — no cycles by construction)."""
    chosen_descs = rng.sample(REQUEST_DESCRIPTIONS, n)
    out: list[Request] = []
    for i in range(n):
        label = LABELS[i]
        desc = chosen_descs[i]
        # 50% chance of having a single dependency on an earlier request.
        deps: tuple[str, ...] = ()
        if i > 0 and rng.random() < 0.5:
            dep_idx = rng.randrange(i)
            deps = (LABELS[dep_idx],)
        priority = rng.choice(PRIORITIES)
        out.append(Request(label=label, description=desc, deps=deps,
                           priority=priority))
    return out


def build_scenario(
    rng, scenario_idx: int, *, n_requests_range=(4, 7), n_updates_range=(0, 3),
) -> TriageScenario:
    n_requests = rng.randint(*n_requests_range)
    if n_requests > len(REQUEST_DESCRIPTIONS):
        n_requests = len(REQUEST_DESCRIPTIONS)
    requests = _build_dag(rng, n_requests)

    # Pick status updates.
    n_updates = rng.randint(*n_updates_range)
    updates: list[StatusUpdate] = []
    updateable = list(range(n_requests))
    rng.shuffle(updateable)
    for k in range(min(n_updates, len(updateable))):
        idx = updateable[k]
        label = requests[idx].label
        new_status = rng.choice(["done", "deprioritized"])
        updates.append(StatusUpdate(label=label, new_status=new_status))

    # Apply updates → final status map.
    final_status = {r.label: r.status for r in requests}
    for u in updates:
        final_status[u.label] = u.new_status

    # Passage order: first one passage per request (in arrival order), then
    # one passage per update. The driver iterates in this order.
    passage_labels = [r.label for r in requests] + [u.label for u in updates]

    return TriageScenario(
        scenario_idx=scenario_idx,
        requests=requests,
        updates=updates,
        final_status=final_status,
        passage_labels=passage_labels,
    )


def is_ready(scen: TriageScenario, label: str) -> bool:
    """A request is ready iff: pending AND all deps are done."""
    if scen.final_status.get(label) != "pending":
        return False
    req = next(r for r in scen.requests if r.label == label)
    return all(scen.final_status.get(d) == "done" for d in req.deps)


def blocking_deps(scen: TriageScenario, label: str) -> list[str]:
    """Labels of deps still blocking `label`."""
    req = next(r for r in scen.requests if r.label == label)
    return [d for d in req.deps if scen.final_status.get(d) != "done"]


def config_space_size() -> int:
    """For n=5 requests: C(24, 5) descriptions × 2^4 dep edges (each request
    after first chooses 0 or 1 dep) × 3^5 priorities × ~5 updates each w/ 2 status."""
    n = 5
    n_descs = 1
    for k in range(n):
        n_descs *= len(REQUEST_DESCRIPTIONS) - k
    n_deps = 2 ** (n - 1)
    n_priorities = len(PRIORITIES) ** n
    n_updates = (n * 2) ** 2     # 2 updates avg, each can be any request × status
    return n_descs * n_deps * n_priorities * n_updates
