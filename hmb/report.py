"""Compact, strictly finite score summaries of a scorer report (``<S>/scores.json``).

``hmb.scoring.score_session`` writes the detailed report; :func:`summarize_report` reduces it to
what the command line (``hmb.cli``), ``humanscore.json`` (``hmb.pipeline``) and the
multi-person summary (``hmb.multi``) show::

    {"scores": {six metrics}, "tiers": {anatomy, kinematic, kinetic}, "overall",
     "status": {metric: ok | unsupported | missing_input | error},
     "reasons": {metric: one line, for every metric that is not ok},
     "errors": {metric: full error text},
     "paper_scores": {"scores", "tiers", "overall", "status", "reasons"}   (only when computed)}

:func:`format_scores` turns that summary into a short console table and :func:`reason_lines`
lists why a metric has no score.  Every score in a summary is a finite float or ``None``.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from hmb.scoring.score_session import METRICS, TIERS, status_reason

#: Display names of the six metrics, in the paper's order.
LABELS = {"extra_limbs": "extra limbs", "bone_length": "bone length", "joint_range": "joint range",
          "self_collision": "self-collision", "kinematic_extremes": "kinematic extremes",
          "motion_smoothness": "motion smoothness"}


def _score(value) -> Optional[float]:
    """``value`` as a float when it is a finite number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _section(metrics: dict, tiers: Optional[dict], overall, error: Optional[str]) -> dict:
    """Summary of one set of six metric entries (the HumanScore scores or the paper scores).

    A metric absent from ``metrics`` is an ``error`` (with ``error`` as its reason when given);
    an ``ok`` entry without a finite score is downgraded to ``error``.
    """
    scores, status, reasons, errors = {}, {}, {}, {}
    for name in METRICS:
        entry = metrics.get(name)
        if entry is None:
            scores[name], status[name] = None, "error"
            errors[name] = error or "not scored"
            reasons[name] = status_reason(name, {"status": "error", "error": errors[name]})
            continue
        scores[name] = _score(entry.get("score"))
        state = entry.get("status", "error")
        status[name] = "error" if state == "ok" and scores[name] is None else state
        if entry.get("error"):
            errors[name] = entry["error"]
        if status[name] != "ok":
            reasons[name] = ("the report has no finite score" if state == "ok"
                             else status_reason(name, dict(entry, status=state)) or state)
    complete = all(status[name] == "ok" for name in METRICS)
    tiers = {tier: _score((tiers or {}).get(tier)) for tier in TIERS}
    return {"scores": scores, "tiers": tiers, "overall": _score(overall) if complete else None,
            "status": status, "reasons": reasons, "errors": errors}


def summarize_report(report: Optional[dict], *, error: Optional[str] = None) -> dict:
    """The summary of a ``scores.json`` report (see the module docstring).

    ``error`` (scoring failed, no report) marks every metric missing from the report as
    ``error`` with that reason.  ``overall`` is null unless all six metrics are ``ok``.
    """
    report = report or {}
    summary = _section(report.get("metrics") or {}, report.get("tiers"), report.get("overall"), error)
    paper = report.get("paper_scores")
    if isinstance(paper, dict):
        section = _section(paper.get("metrics") or {}, paper.get("tiers"), paper.get("overall"), None)
        summary["paper_scores"] = {key: section[key] for key in ("scores", "tiers", "overall", "status", "reasons")}
    return summary


def is_complete(summary: dict) -> bool:
    """All six scores (and the six paper scores, when present) were computed."""
    paper = summary.get("paper_scores")
    return (all(state == "ok" for state in (summary.get("status") or {}).values())
            and len(summary.get("status") or {}) == len(METRICS)
            and (paper is None or all(state == "ok" for state in (paper.get("status") or {}).values())))


def _fmt(value) -> str:
    """A score as a 6-character table cell (``-`` when there is none)."""
    return f"{value:6.1f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "     -"


def format_scores(summary: dict, title: str) -> str:
    """Table of a summary: one row per metric (score, paper score when present, status) and the tiers."""
    paper = summary.get("paper_scores")
    header = f"  {'tier':10s} {'metric':20s} {'score':>6s}" + (f" {'paper':>6s}" if paper else "") + "  status"
    lines = [title, "", header, "  " + "-" * (len(header) - 2)]
    for tier, pair in TIERS.items():
        for k, metric in enumerate(pair):
            row = f"  {tier if k == 0 else '':10s} {LABELS[metric]:20s} {_fmt(summary['scores'].get(metric))}"
            if paper:
                row += f" {_fmt(paper['scores'].get(metric))}"
            lines.append(row + f"  {summary['status'].get(metric)}")
    lines.append("  " + "-" * (len(header) - 2))
    for tier in TIERS:
        row = f"  {'':10s} {tier + ' tier':20s} {_fmt(summary['tiers'].get(tier))}"
        if paper:
            row += f" {_fmt(paper['tiers'].get(tier))}"
        lines.append(row)
    row = f"  {'':10s} {'OVERALL':20s} {_fmt(summary.get('overall'))}"
    if paper:
        row += f" {_fmt(paper.get('overall'))}"
    lines.append(row)
    return "\n".join(lines)


def reason_lines(summary: dict) -> list:
    """``"<metric> <status>: <reason>"`` for every metric that is not ok (paper ones as ``"paper <metric> ..."``)."""
    lines = []
    for prefix, section in (("", summary), ("paper ", summary.get("paper_scores") or {})):
        reasons = section.get("reasons") or {}
        for metric, state in (section.get("status") or {}).items():
            if state != "ok" and reasons.get(metric):
                lines.append(f"{prefix}{metric} {state}: {reasons[metric]}")
    return lines


def json_safe(value: Any):
    """Copy of ``value`` with non-finite floats as null (reports are written with ``allow_nan=False``)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
