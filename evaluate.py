#!/usr/bin/env python3
"""AgentVidBench evaluation entry point.

Takes one experiment dir produced by `inference.py` and writes
`<run>/evaluation/{letters,judge,results,summary.json}`.

Quick example
-------------
    python evaluate.py exp/ours_gemini-2.5-flash_run1/

Pipeline (per qid, parallel):
  1. Letter extraction — N-vote majority via Vertex Gemini Flash.
  2. Process scoring   — P1-P5 + milestone coverage via the chosen judge
                          provider (default: Anthropic claude-opus-4-7).
  3. Combined per-Q result joining the predicted letter with the dataset's
     gold answer and the judge output.

Aggregation across the run: accuracy %, P-axis means, by_difficulty /
by_category / by_skill slices → `<run>/evaluation/summary.json`.

Resume: re-running skips per-Q files that already exist under
`<run>/evaluation/results/`. Pass `--rerun` to re-evaluate every question.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.aggregate import aggregate_run  # noqa: E402
from evaluation.extract_letters import extract_one_record  # noqa: E402
from evaluation.judge import judge_one  # noqa: E402
from framework.orchestration import DEFAULT_QUESTIONS_JSONL  # noqa: E402


def _load_questions(questions_jsonl: Path) -> dict[int, dict]:
    """Read raw HF questions.jsonl. Evaluation needs the full row
    (milestones, ecom_required, answer_explanation, options, skills,
    categories, difficulty, trajectory) — load_items() drops those fields
    because inference doesn't need them.
    """
    out: dict[int, dict] = {}
    with open(questions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["question_id"]] = row
    return out


def _parse_qids(spec: str | None, all_qids: list[int]) -> list[int]:
    if not spec:
        return all_qids
    out = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    return sorted(out & set(all_qids))


def _preflight_or_die(args) -> None:
    # Letter extraction always uses Vertex Gemini Flash.
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        sys.exit("ERROR: GOOGLE_CLOUD_PROJECT not set "
                 "(letter extraction uses Vertex Gemini Flash).")
    if args.judge_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set "
                 "(required for --judge-provider anthropic).")
    if args.judge_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set "
                 "(required for --judge-provider openai).")


def _process_one(framework: str, qid: int, q: dict, run_dir: Path,
                 args) -> tuple[int, dict]:
    """Run letter extraction + judge for one qid, write per-stage artifacts,
    return the joined result record. Caches per-stage outputs so a partial
    failure can resume cheaply."""
    eval_dir = run_dir / "evaluation"
    letters_dir = eval_dir / "letters"
    judge_dir = eval_dir / "judge"
    results_dir = eval_dir / "results"
    for d in (letters_dir, judge_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1. Letter extraction (cached)
    letters_path = letters_dir / f"question{qid}.json"
    if letters_path.exists() and not args.rerun:
        letters = json.loads(letters_path.read_text(encoding="utf-8"))
    else:
        letters = extract_one_record(framework, qid, run_dir, n_runs=args.n_extract)
        letters_path.write_text(json.dumps(letters, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    pred_letter = letters.get("majority_letter")

    # 2. Process scoring (cached)
    judge_path = judge_dir / f"question{qid}.json"
    if judge_path.exists() and not args.rerun:
        judge_rec = json.loads(judge_path.read_text(encoding="utf-8"))
    else:
        judge_rec = judge_one(qid, framework, q, run_dir, pred_letter,
                              provider=args.judge_provider)
        judge_path.write_text(json.dumps(judge_rec, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # 3. Combined result
    result = {
        "question_id": qid,
        "framework": framework,
        "model": args._model,  # set by main()
        "accuracy": judge_rec.get("accuracy"),
        "process": judge_rec.get("process") or {},
        "judge_model": judge_rec.get("judge_model"),
        "letter_extraction": {
            "majority": pred_letter,
            "agreement": letters.get("agreement"),
            "n_null": letters.get("n_null"),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    if "error" in judge_rec:
        result["error"] = judge_rec["error"]
    results_path = results_dir / f"question{qid}.json"
    results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return qid, result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AgentVidBench evaluation — score one inference run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dir", type=Path,
                   help="Path to an experiment dir (e.g. exp/ours_gemini-2.5-flash_run1/).")
    p.add_argument("--questions", default=None,
                   help="Comma-separated qids or ranges ('1,5,10-20'). Default: all in the run.")
    p.add_argument("--workers", type=int, default=16,
                   help="Thread pool size for parallel per-Q evaluation.")
    p.add_argument("--n-extract", type=int, default=8,
                   help="Letter-extraction votes (Gemini Flash candidate_count).")
    p.add_argument("--judge-provider", default="anthropic",
                   choices=["anthropic", "openai", "gemini_vertex"],
                   help="LLM judge provider for P1-P5 scoring.")
    p.add_argument("--rerun", action="store_true",
                   help="Re-evaluate every qid (default: skip qids whose results/question*.json exists).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan, don't call APIs.")
    p.add_argument("--questions-jsonl", default=str(DEFAULT_QUESTIONS_JSONL))
    return p


def main():
    args = build_parser().parse_args()

    run_dir: Path = args.run_dir
    inf_summary_path = run_dir / "inference" / "summary.json"
    if not inf_summary_path.exists():
        sys.exit(f"ERROR: inference summary not found at {inf_summary_path}. "
                 f"Did you run `python inference.py ... --tag {run_dir.name}` first?")

    inf_summary = json.loads(inf_summary_path.read_text(encoding="utf-8"))
    framework = inf_summary["experiment"]
    model = inf_summary["model"]
    tag = inf_summary["tag"]
    args._model = model

    _preflight_or_die(args)

    questions = _load_questions(Path(args.questions_jsonl))

    # Qids actually present in this run (skip any inference-errored Qs).
    inference_qids = sorted(
        r["question_id"] for r in inf_summary.get("results", [])
        if "error" not in r
    )
    qids = _parse_qids(args.questions, inference_qids)

    # Resume: drop qids whose results/question*.json already exists (unless --rerun).
    results_dir = run_dir / "evaluation" / "results"
    if not args.rerun and results_dir.exists():
        already = {int(p.stem[len("question"):]) for p in results_dir.glob("question*.json")}
        skipped = [q for q in qids if q in already]
        qids = [q for q in qids if q not in already]
        if skipped:
            print(f"Resume: {len(skipped)} qids already evaluated, skipping.")

    print(f"=== evaluate ===")
    print(f"Run:    {run_dir}")
    print(f"FW:     {framework}  Model: {model}  Tag: {tag}")
    print(f"Qids:   {len(qids)} pending  Workers: {args.workers}")
    print(f"Judge:  {args.judge_provider}  Letter votes: {args.n_extract}")
    print("=" * 50)

    if args.dry_run:
        print(f"(dry-run: would evaluate {len(qids)} qids)")
        return

    if qids:
        t0 = time.time()
        done = err = 0
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_process_one, framework, qid, questions[qid], run_dir, args): qid
                    for qid in qids}
            for fut in cf.as_completed(futs):
                qid = futs[fut]
                try:
                    _, rec = fut.result()
                    is_err = "error" in rec
                    err += int(is_err)
                    flag = "[ERR]" if is_err else "[ok]"
                    correct = (rec.get("accuracy") or {}).get("correct")
                    pred = (rec.get("letter_extraction") or {}).get("majority")
                    gold = (rec.get("accuracy") or {}).get("gold")
                    print(f"  {flag} q{qid}: pred={pred} gold={gold} correct={correct} "
                          f"elapsed={rec.get('elapsed_s')}s")
                except Exception as e:
                    err += 1
                    print(f"  [ERR] q{qid}: executor exception: {e!r}")
                done += 1
        print(f"\nEvaluated {done} qids in {time.time() - t0:.1f}s, errors={err}")

    # Aggregate
    summary = aggregate_run(run_dir, questions, experiment=framework, model=model, tag=tag)
    print(f"\nSummary: {run_dir / 'evaluation' / 'summary.json'}")
    print(f"  total={summary['n']}, n_correct={summary['n_correct']}, "
          f"accuracy={summary['accuracy']}")
    means = summary["process_means"]
    print(f"  process_means: P1={means['P1']} P2={means['P2']} P3={means['P3']} "
          f"P4={means['P4']} P5={means['P5']}  traj={means['traj']}  mc_rate={means['mc_rate']}")


if __name__ == "__main__":
    main()
