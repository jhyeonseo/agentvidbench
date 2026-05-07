"""Evaluation pipeline — letter extraction, LLM judge, aggregation.

Used by the top-level `evaluate.py` CLI to score the outputs of an inference
run (`exp/<run>/inference/`) against the dataset's ground-truth answers and
the curated milestones (P1-P5 process rubric).

Modules:
  - extract_letters: N-vote LLM extraction of the predicted A-Z letter
                     from a model's raw output (Vertex Gemini Flash).
  - judge:           process scoring (P1-P5 + milestone coverage) via an
                     LLM judge (Anthropic / OpenAI / Vertex Gemini).
  - aggregate:       roll per-Q results up into evaluation/summary.json.

Per-run outputs land under `exp/<run>/evaluation/`:
  letters/question<N>.json — vote ledger + majority letter
  judge/question<N>.json   — process scores + milestone coverage
  results/question<N>.json — combined per-Q record (accuracy + process)
  summary.json             — aggregate metrics (accuracy, P-axis means, ...)
"""
