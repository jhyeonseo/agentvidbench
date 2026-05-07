"""Aggregate per-question evaluation results into evaluation/summary.json.

Reads `<run>/evaluation/results/question*.json`, joins on the dataset's
difficulty/categories/skills slices, computes accuracy + P-axis means.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


def _mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 3) if xs else None


def _accumulate(records: list[dict]) -> dict:
    total = len(records)
    n_correct = sum(1 for r in records if (r.get("accuracy") or {}).get("correct"))
    accuracy = (n_correct / total) if total else None

    p_axes = {axis: [] for axis in ("P1", "P2", "P3", "P4", "P5")}
    trajs, mc_rates = [], []
    for r in records:
        proc = r.get("process") or {}
        scores = proc.get("scores") or {}
        for axis in p_axes:
            p_axes[axis].append(scores.get(axis))  # may be None
        trajs.append(proc.get("traj"))
        mc_rates.append(proc.get("mc_rate"))

    process_means = {axis: _mean_or_none(vals) for axis, vals in p_axes.items()}
    process_means["traj"] = _mean_or_none(trajs)
    process_means["mc_rate"] = _mean_or_none(mc_rates)

    return {
        "n": total,
        "n_correct": n_correct,
        "accuracy": round(accuracy, 3) if accuracy is not None else None,
        "process_means": process_means,
    }


def _slice(records: list[dict], questions: dict[int, dict],
           field: str, listy: bool) -> dict:
    """Group records by `field` from the question. If listy, a record
    contributes to every value in the list (e.g. categories=[a,b,c])."""
    buckets: dict[str, list[dict]] = {}
    for r in records:
        qid = r["question_id"]
        q = questions.get(qid)
        if not q:
            continue
        v = q.get(field)
        if listy:
            for item in (v or []):
                buckets.setdefault(item, []).append(r)
        else:
            if v is not None:
                buckets.setdefault(str(v), []).append(r)
    return {k: _accumulate(v) for k, v in sorted(buckets.items())}


def aggregate_run(run_dir: Path, questions: dict[int, dict],
                  experiment: str, model: str, tag: str) -> dict:
    """Read evaluation/results/question*.json under run_dir, compute summary.

    Caller passes `experiment`/`model`/`tag` (typically read from the
    inference summary) so the eval summary stays self-describing.
    """
    results_dir = run_dir / "evaluation" / "results"
    records = []
    for f in sorted(results_dir.glob("question*.json")):
        try:
            records.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue

    overall = _accumulate(records)
    summary = {
        "experiment": experiment,
        "model": model,
        "tag": tag,
        **overall,
        "by_difficulty": _slice(records, questions, "difficulty", listy=False),
        "by_category":   _slice(records, questions, "categories", listy=True),
        "by_skill":      _slice(records, questions, "skills", listy=True),
    }
    out = run_dir / "evaluation" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
