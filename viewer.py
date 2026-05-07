"""AgentVidBench viewer.

Browses two things in a local web UI::

    streamlit run viewer.py

  - the dataset itself (``dataset/questions.jsonl`` + ``dataset/videos/`` +
    ``dataset/transcripts/``) — question text, options, video preview,
    ground-truth milestones, curated answer explanation and reasoning trajectory.
  - one or more eval runs under ``exp/<run>/evaluation/{summary.json,
    results/question*.json}`` — cross-run summary metrics + side-by-side per-Q
    cards (scores, rationales, milestone coverage, failure tags).

Either source is optional. With no dataset, per-Q metadata is replaced by a
"dataset not downloaded" notice; with no runs, only the dataset browser shows.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
EXP_DIR = REPO_ROOT / "exp"
DATASET_DIR = REPO_ROOT / "dataset"

STATUS_EMOJI = {"covered": "✓", "partial": "◐", "incorrect": "✗", "missing": "·"}
PALETTE = ["#6366f1", "#10b981", "#f59e0b", "#ec4899", "#0ea5e9", "#8b5cf6"]

# Stored JSON keys are P1..P5; the viewer surfaces them as mnemonics.
P_AXES = ["P1", "P2", "P3", "P4", "P5"]
P_LABEL = {
    "P1": "TU",     # task understanding
    "P2": "ECov",   # evidence coverage
    "P3": "EG",     # evidence grounding
    "P4": "ECom",   # exhaustive sweep (e-commerce-required questions)
    "P5": "RF",     # reasoning faithfulness
}

CSS_PATH = REPO_ROOT / "viewer.css"

# Light-theme radar colors. Mirrors the corresponding CSS vars in viewer.css
# (--text, --text-subtle, --border, --border-strong); duplicated here because
# Plotly figures take colors as arguments, not CSS classes.
RADAR_COLORS = {
    "text":  "#334155",
    "muted": "#94a3b8",
    "grid":  "#e2e8f0",
    "line":  "#cbd5e1",
}


@st.cache_data
def _load_css_file() -> str:
    try:
        return CSS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def inject_styles() -> None:
    st.markdown(f"<style>\n{_load_css_file()}\n</style>", unsafe_allow_html=True)


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _esc(s) -> str:
    return html.escape("" if s is None else str(s))


def _run_color(run_name: str, all_names: list[str]) -> str:
    try:
        idx = all_names.index(run_name)
    except ValueError:
        idx = 0
    return PALETTE[idx % len(PALETTE)]


def _run_chips_html(run: dict) -> str:
    s = run.get("summary") or {}
    parts = []
    if s.get("experiment"):
        parts.append(f'<span class="chip chip-fw">{_esc(s["experiment"])}</span>')
    if s.get("model"):
        parts.append(f'<span class="chip chip-model">{_esc(s["model"])}</span>')
    if s.get("tag"):
        parts.append(f'<span class="chip chip-tag">{_esc(s["tag"])}</span>')
    return "".join(parts) or f'<span class="chip">{_esc(run["name"])}</span>'


def _short_name(run: dict) -> str:
    """Human-readable column header: framework/model (skip tag for compactness)."""
    s = run.get("summary") or {}
    if s.get("experiment") and s.get("model"):
        return f"{s['experiment']}/{s['model']}"
    return run["name"]


@st.cache_data
def discover_runs() -> list[dict]:
    """Find every ``exp/<run>/evaluation/summary.json`` and return run dicts.

    Each dict: ``{name, summary, qids, results_dir}``. ``qids`` is the sorted
    list of question_ids that have a per-Q results file on disk.
    """
    runs: list[dict] = []
    if not EXP_DIR.exists():
        return runs
    for run_dir in sorted(p for p in EXP_DIR.iterdir() if p.is_dir()):
        summary_path = run_dir / "evaluation" / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as ex:
            runs.append({"name": run_dir.name, "summary": None, "qids": [],
                         "results_dir": run_dir / "evaluation" / "results",
                         "error": f"failed to parse summary.json: {ex}"})
            continue
        results_dir = run_dir / "evaluation" / "results"
        qids: list[int] = []
        if results_dir.exists():
            for f in results_dir.glob("question*.json"):
                stem = f.stem  # "question1"
                try:
                    qids.append(int(stem[len("question"):]))
                except ValueError:
                    pass
        runs.append({"name": run_dir.name, "summary": summary,
                     "qids": sorted(qids), "results_dir": results_dir})
    return runs


@st.cache_data
def load_result(run_name: str, qid: int) -> dict | None:
    path = EXP_DIR / run_name / "evaluation" / "results" / f"question{qid}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as ex:
        return {"_error": f"failed to parse {path.name}: {ex}"}


def _slice_rows(runs: list[dict], slice_key: str) -> list[tuple[str, list[str]]]:
    """Build [(slice_value, [cell_per_run])] for one slice axis. Run column
    order matches the input list, so the rendered table aligns with the
    radars and overall-cards above it."""
    all_keys: set[str] = set()
    per_run: dict[str, dict] = {}
    for r in runs:
        s = (r.get("summary") or {}).get(slice_key) or {}
        per_run[r["name"]] = s
        all_keys.update(s.keys())
    rows: list[tuple[str, list[str]]] = []
    for k in sorted(all_keys):
        cells = []
        for r in runs:
            cell = per_run[r["name"]].get(k)
            if not cell:
                cells.append("—")
            else:
                acc = cell.get("accuracy")
                nc, n = cell.get("n_correct"), cell.get("n")
                pct = f"{acc * 100:.0f}%" if isinstance(acc, (int, float)) else "—"
                cells.append(f"{pct} ({nc}/{n})")
        rows.append((k, cells))
    return rows


def render_slice_table(runs: list[dict], slice_key: str) -> str:
    """Theme-aware HTML table for a slice axis (.avb-slice-table styled in CSS).

    `st.dataframe` is canvas-based and can't read our CSS variables, so the
    by_difficulty / by_category / by_skill tables looked unthemed. Plain HTML
    inherits the page tokens and stays aligned with the chips/pills above."""
    rows = _slice_rows(runs, slice_key)
    if not rows:
        return ""
    headers = ["".join(("<th></th>",))] + [
        f'<th><span class="run-dot" style="background:{_run_color(r["name"], [x["name"] for x in runs])};"></span>'
        f'{_esc(_short_name(r))}</th>'
        for r in runs
    ]
    head = f'<thead><tr>{"".join(headers)}</tr></thead>'
    body_rows = []
    for k, cells in rows:
        tds = "".join(f'<td>{_esc(c)}</td>' for c in cells)
        body_rows.append(f'<tr><td class="row-label">{_esc(k)}</td>{tds}</tr>')
    body = f'<tbody>{"".join(body_rows)}</tbody>'
    return f'<table class="avb-slice-table">{head}{body}</table>'


def render_overall_cards(runs: list[dict]) -> None:
    all_names = [r["name"] for r in runs]
    for run in runs:
        s = run.get("summary") or {}
        color = _run_color(run["name"], all_names)
        with st.container(border=True):
            st.markdown(
                f'<div class="run-accent" style="background:{color};"></div>'
                f'{_run_chips_html(run)}',
                unsafe_allow_html=True,
            )
            cols = st.columns([1.3, 1, 1, 1, 1, 1, 1, 1])
            acc = s.get("accuracy")
            nc, n = s.get("n_correct"), s.get("n")
            cols[0].metric(
                "Accuracy",
                f"{acc * 100:.0f}%" if isinstance(acc, (int, float)) else "—",
                delta=f"{nc} of {n}" if nc is not None and n is not None else None,
                delta_color="off",
            )
            pm = s.get("process_means") or {}
            for col, key in zip(cols[1:], P_AXES + ["traj", "mc_rate"]):
                col.metric(P_LABEL.get(key, key), _fmt(pm.get(key)))


def _hex_to_rgba(color: str, alpha: float) -> str:
    """`#rrggbb` + alpha → `rgba(r, g, b, a)` (plotly rejects 8-digit hex)."""
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _radar_figure(run: dict, color: str, palette: dict[str, str]) -> tuple[go.Figure, bool]:
    pm = (run.get("summary") or {}).get("process_means") or {}
    raw = [pm.get(a) for a in P_AXES]
    has_null = any(v is None for v in raw)
    values = [0 if v is None else v for v in raw]
    labels = [P_LABEL[a] for a in P_AXES]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],          # close the polygon
        theta=labels + [labels[0]],
        fill="toself",
        line=dict(color=color, width=2),
        fillcolor=_hex_to_rgba(color, 0.2),
        marker=dict(size=5, color=color),
        name=run["name"],
        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
    ))
    font_family = ('ui-sans-serif, system-ui, -apple-system, "Segoe UI", '
                   'Roboto, "Helvetica Neue", sans-serif')
    fig.update_layout(
        font=dict(family=font_family, size=11, color=palette["text"]),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 2], dtick=0.5,
                            gridcolor=palette["grid"], linecolor=palette["line"],
                            tickfont=dict(family=font_family, size=9,
                                          color=palette["muted"])),
            angularaxis=dict(gridcolor=palette["grid"], linecolor=palette["line"],
                             tickfont=dict(family=font_family, size=12,
                                           color=palette["text"])),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=24, r=24, t=12, b=12),
        height=280,
    )
    return fig, has_null


def render_radars(runs: list[dict]) -> None:
    """One TU/ECov/EG/ECom/RF radar per run. null axes (anchored ECom) -> 0."""
    all_names = [r["name"] for r in runs]
    palette = RADAR_COLORS
    cols = st.columns(len(runs))
    any_null = False
    for col, run in zip(cols, runs):
        color = _run_color(run["name"], all_names)
        fig, has_null = _radar_figure(run, color, palette)
        any_null = any_null or has_null
        with col:
            st.markdown(
                f'<div style="text-align:center;font-size:0.85rem;'
                f'color:var(--text-muted);margin-bottom:0.25rem;">'
                f'<span style="color:{color};">●</span> {_esc(_short_name(run))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, width="stretch", key=f"radar_{run['name']}")
    notes = ["raw scores · max **2** per axis"]
    if any_null:
        notes.append("null axes (e.g. ECom on anchored questions) plotted as 0")
    st.caption("  ·  ".join(notes))


def render_summary_compare(runs: list[dict]) -> None:
    st.markdown("##### Overall metrics")
    render_overall_cards(runs)

    st.markdown("##### Process radar (TU · ECov · EG · ECom · RF)")
    render_radars(runs)

    for slice_key, label in [
        ("by_difficulty", "By difficulty"),
        ("by_category",   "By category"),
        ("by_skill",      "By skill"),
    ]:
        html_table = render_slice_table(runs, slice_key)
        if not html_table:
            continue
        st.markdown(f"##### {label}")
        st.markdown(html_table, unsafe_allow_html=True)


@st.cache_data
def _load_dataset() -> dict | None:
    """Read dataset/questions.jsonl + dataset/videos.jsonl. None if missing."""
    qpath = DATASET_DIR / "questions.jsonl"
    vpath = DATASET_DIR / "videos.jsonl"
    if not qpath.exists() or not vpath.exists():
        return None
    questions: dict[int, dict] = {}
    with open(qpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            questions[row["question_id"]] = row
    videos: dict[str, dict] = {}
    with open(vpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            videos[row["file_name"]] = row
    return {"questions": questions, "videos": videos}


def _difficulty_pill_class(diff: str | None) -> str:
    return {"easy": "pill-2", "medium": "pill-1", "hard": "pill-0"}.get(diff or "", "pill-null")


def render_question_panel(qid: int, dataset: dict) -> None:
    q = dataset["questions"].get(qid)
    if not q:
        st.warning(f"q{qid} not in dataset/questions.jsonl")
        return

    with st.container(border=True):
        st.markdown(
            f'<h3 style="margin:0 0 0.4rem 0;">{_esc(q.get("title") or f"q{qid}")}</h3>',
            unsafe_allow_html=True,
        )

        chips = [
            f'<span class="pill-label">difficulty</span>'
            f'<span class="pill {_difficulty_pill_class(q.get("difficulty"))}">'
            f'{_esc(q.get("difficulty") or "—")}</span>',
        ]
        if q.get("ecom_required") is not None:
            chips.append(
                f'<span class="pill-label">sweep</span>'
                f'<span class="pill pill-info">'
                f'{"required" if q.get("ecom_required") else "anchored"}</span>'
            )
        for cat in (q.get("categories") or []):
            chips.append(f'<span class="chip">{_esc(cat)}</span>')
        for skill in (q.get("skills") or []):
            chips.append(f'<span class="chip chip-fw">{_esc(skill)}</span>')
        st.markdown(f'<div style="margin-bottom:0.8rem;">{"".join(chips)}</div>',
                    unsafe_allow_html=True)

        left, right = st.columns([1, 1.2])

        with left:
            video_rel = q.get("video_path")
            video_abs = (DATASET_DIR / video_rel) if video_rel else None
            if video_abs and video_abs.exists():
                st.video(str(video_abs))
            else:
                st.caption(f"(video file not found: {video_rel})")

            video_meta = (dataset["videos"] or {}).get(video_rel) or {}
            if video_meta:
                meta_keys = ("duration", "genre", "length_category",
                             "license", "source", "author", "url")
                lines = []
                for k in meta_keys:
                    v = video_meta.get(k)
                    if v:
                        lines.append(
                            f'<span class="pill-label">{k}</span> {_esc(str(v))}'
                        )
                if video_meta.get("title"):
                    lines.insert(0, f'<b>{_esc(video_meta["title"])}</b>')
                st.markdown(
                    f'<div class="video-meta">' + "<br>".join(lines) + '</div>',
                    unsafe_allow_html=True,
                )

            transcript_rel = q.get("transcript_path")
            tpath = (DATASET_DIR / transcript_rel) if transcript_rel else None
            if tpath and tpath.exists() and tpath.stat().st_size > 0:
                with st.expander("Transcript (.srt)", expanded=False):
                    st.code(tpath.read_text(encoding="utf-8", errors="replace"),
                            language="text")

        with right:
            st.markdown(
                f'<div style="font-size:0.92rem;line-height:1.55;margin-bottom:0.5rem;">'
                f'{_esc(q.get("question_text", ""))}</div>',
                unsafe_allow_html=True,
            )

            gold = q.get("answer")
            opts_html = ['<div style="display:grid;grid-template-columns:1fr 1fr;'
                         'gap:0.25rem 0.5rem;margin-top:0.25rem;">']
            for opt in (q.get("options") or []):
                letter = opt.get("letter", "?")
                text = opt.get("text", "")
                cls = "option-row gold" if letter == gold else "option-row"
                opts_html.append(
                    f'<div class="{cls}">'
                    f'<span class="letter">{_esc(letter)}</span>'
                    f'<span>{_esc(text)}</span></div>'
                )
            opts_html.append("</div>")
            st.markdown("".join(opts_html), unsafe_allow_html=True)
            if q.get("options_type"):
                st.caption(f"options_type: {q['options_type']}")

        milestones = q.get("milestones") or []
        if milestones:
            with st.expander(f"Ground-truth milestones ({len(milestones)})",
                             expanded=False):
                for m in milestones:
                    st.markdown(
                        f"**{m.get('id', '?')}** · _{m.get('type', '?')}_ — "
                        f"{m.get('description', '')}"
                    )

        explanation = q.get("answer_explanation")
        if explanation:
            with st.expander("Answer explanation", expanded=False):
                st.markdown(explanation.replace("\n", "  \n"))

        traj = q.get("trajectory")
        if isinstance(traj, dict) and traj.get("steps"):
            n = len(traj["steps"])
            with st.expander(f"Curated reasoning trajectory ({n} steps)",
                             expanded=False):
                if traj.get("final_answer"):
                    st.markdown(f"**Final answer**: {traj['final_answer']}")
                st.json(traj)


def _score_pill_html(label: str, value) -> str:
    if value is None:
        cls = "pill-null"
        text = "—"
    elif isinstance(value, (int, float)) and value in (0, 1, 2):
        cls = f"pill-{int(value)}"
        text = str(int(value))
    else:
        cls = "pill-null"
        text = _fmt(value)
    return (f'<span class="pill-label">{label}</span>'
            f'<span class="pill {cls}">{text}</span>')


def _render_per_q_card(run: dict, qid: int, color: str) -> None:
    with st.container(border=True):
        st.markdown(
            f'<div class="run-accent" style="background:{color};"></div>'
            f'{_run_chips_html(run)}',
            unsafe_allow_html=True,
        )
        if qid not in run["qids"]:
            st.caption("(not run)")
            return
        rec = load_result(run["name"], qid)
        if rec is None:
            st.caption("(no per-Q file)")
            return
        if "_error" in rec:
            st.error(rec["_error"])
            return

        acc = rec.get("accuracy") or {}
        gold, pred, correct = acc.get("gold"), acc.get("pred"), acc.get("correct")
        icon = "✅" if correct else "❌"
        ans_cls = "pill-2" if correct else "pill-0"
        st.markdown(
            '<div class="qmeta">'
            f'<span class="icon">{icon}</span>'
            f'<span class="pill-label">gold</span>'
            f'<span class="pill {ans_cls}">{_esc(gold)}</span>'
            f'<span class="pill-label">pred</span>'
            f'<span class="pill {ans_cls}">{_esc(pred)}</span>'
            f'<span class="pill-label">judge</span>'
            f'<code>{_esc(rec.get("judge_model", "?"))}</code>'
            '</div>',
            unsafe_allow_html=True,
        )

        process = rec.get("process") or {}
        scores = process.get("scores") or {}
        score_html = "".join(_score_pill_html(P_LABEL[k], scores.get(k))
                             for k in P_AXES)
        traj = process.get("traj")
        mc = process.get("mc_rate")
        score_html += (
            f'<span class="pill-label">traj</span>'
            f'<span class="pill pill-info">{_fmt(traj)}</span>'
            f'<span class="pill-label">mc_rate</span>'
            f'<span class="pill pill-info">{_fmt(mc)}</span>'
        )
        st.markdown(f'<div style="margin:0.25rem 0 0.75rem;">{score_html}</div>',
                    unsafe_allow_html=True)

        rationales = process.get("rationales") or {}
        if rationales:
            with st.expander("Rationales", expanded=False):
                for axis in P_AXES:
                    if axis in rationales:
                        st.markdown(f"**{P_LABEL[axis]}** — {rationales[axis]}")

        milestones = process.get("milestones") or []
        if milestones:
            with st.expander(f"Milestones ({len(milestones)})", expanded=False):
                rows = [{"id": m.get("id"),
                         "status": f"{STATUS_EMOJI.get(m.get('status'), '?')} "
                                   f"{m.get('status', '')}",
                         "evidence": m.get("evidence", "")} for m in milestones]
                st.dataframe(rows, width="stretch", hide_index=True)

        tags = process.get("failure_tags") or []
        if tags:
            pills = "".join(f'<span class="tag-pill">{_esc(t)}</span>' for t in tags)
            st.markdown(
                f'<div style="margin:0.4rem 0;">'
                f'<span class="pill-label">failure tags</span>{pills}</div>',
                unsafe_allow_html=True,
            )

        summary_text = process.get("summary")
        if summary_text:
            st.markdown(f"> {summary_text}")

        with st.expander("Raw JSON", expanded=False):
            st.json(rec)


def render_per_q_compare(runs: list[dict], qid: int) -> None:
    all_names = [r["name"] for r in runs]
    cols = st.columns(len(runs))
    for col, run in zip(cols, runs):
        with col:
            _render_per_q_card(run, qid, _run_color(run["name"], all_names))


def main() -> None:
    st.set_page_config(
        page_title="AgentVidBench Viewer",
        page_icon="🎬",
        layout="wide",
    )

    inject_styles()
    st.markdown(
        '<div class="avb-hero">'
        '<h1>AgentVidBench <span class="accent">·</span> viewer</h1>'
        '</div>'
        f'<div class="avb-subtitle">{_esc(EXP_DIR)}</div>',
        unsafe_allow_html=True,
    )

    runs = discover_runs()
    dataset = _load_dataset()

    if not runs and dataset is None:
        st.warning(
            f"No `evaluation/summary.json` under `{EXP_DIR}/` and no `dataset/`. "
            f"Run `python evaluate.py exp/<run>/` and/or download the dataset "
            f"(see README §2)."
        )
        return

    selected_runs: list[dict] = []
    if runs:
        selected = st.multiselect(
            "Runs to compare (max 3)",
            options=[r["name"] for r in runs],
            default=[],
            max_selections=3,
            format_func=lambda n: next(_short_name(r) for r in runs if r["name"] == n),
        )
        selected_runs = [r for r in runs if r["name"] in selected]
    else:
        st.info(f"No runs found under `{EXP_DIR}/`. Per-run comparison disabled.")

    summary_tab, per_q_tab = st.tabs(["Summary", "Per-Question"])

    with summary_tab:
        if selected_runs:
            render_summary_compare(selected_runs)
        else:
            st.info("Select one or more runs above to see summary metrics.")

    with per_q_tab:
        run_qids = sorted({q for r in selected_runs for q in r["qids"]})
        dataset_qids = sorted(dataset["questions"].keys()) if dataset else []
        all_qids = sorted(set(run_qids) | set(dataset_qids))

        if not all_qids:
            st.info("No questions to display.")
            return

        qid = st.selectbox("Question", options=all_qids,
                           format_func=lambda q: f"q{q}")

        if dataset is None:
            st.warning(
                f"`dataset/` not downloaded — see README §2 to populate it. "
                f"Question metadata, video preview, and ground-truth milestones "
                f"are unavailable."
            )
        else:
            render_question_panel(qid, dataset)

        if selected_runs:
            st.markdown("##### Model outputs")
            render_per_q_compare(selected_runs, qid)
        elif runs:
            st.caption("Select runs above to compare model outputs on this question.")


if __name__ == "__main__":
    main()
