"""JsonPromptPlanner — prompt-only JSON tool calling (tool-as-key schema).

Bypasses vLLM's native tool_call parsing (which has known bugs for Qwen3
Thinking #39056 and Gemma 4 #39468) by:
  - NOT passing tools=[] / tool_choice="auto" to chat.completions.create
  - Embedding tool descriptions + JSON schema into system_prompt as text
  - Extracting tool calls from model's content via robust JSON parser
  - Appending tool results as role:user text (not role:tool with tool_call_id)

Schema (flat tool-as-key — minimizes brace-counting load on small models):

  Tool call:    {"thought": "...", "<tool_name>": {<arg>: <val>, ...}}      # 2-level
  Final answer: {"thought": "...", "answer": "<A-Z letter>"}                 # 1-level

The top-level key after "thought" IS the chosen action — semantically
mirrors AVP's flat MCQ_SCHEMA / array-of-flat-step PLAN_SCHEMA pattern,
which we verified is robust on Qwen 9B / Gemma E*B (smoke v2 raw capture).

react.py sees the same PlannerResponse shape — synthesized PlannerToolCall
objects with deterministic fake call_ids. The ReAct loop is unchanged.

Use only for model families with broken native tool parser (Qwen3 / Gemma 4).
GPT-5 / Claude / Gemini keep the OpenAICompatiblePlanner path with native
tools=[] API since their parsers work correctly.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Sequence

from framework.models.base import (
    PlannerInput,
    PlannerResponse,
    PlannerToolCall,
)
from framework.models.openai_compatible.planner import OpenAICompatiblePlanner

logger = logging.getLogger(__name__)


def _parse_json_block(text: str) -> Optional[dict]:
    """Robust 5-stage JSON extraction. Mirrors AVP's parse_json_response.

    1. Strip ``<think>...</think>`` prefix (so JSON inside thinking is ignored
       — model echoes are common).
    2. Direct json.loads on stripped text.
    3. Markdown fenced block (json or generic) — pick LAST parseable match.
    4. Top-level balanced-brace scan ``{...}`` — pick LAST valid candidate.
    5. JSON repair: append missing trailing ``}`` for unclosed brace dangle
       (catches small models truncating before closing).
    """
    if not text:
        return None
    after_think = text
    last_close = text.rfind("</think>")
    if last_close != -1:
        after_think = text[last_close + len("</think>"):]
    try:
        return json.loads(after_think.strip())
    except json.JSONDecodeError:
        pass
    for pat in [r"```json\s*\n(.*?)\n[ \t]*```", r"```\s*\n(.*?)\n[ \t]*```"]:
        for cand in reversed(re.findall(pat, after_think, re.DOTALL)):
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                continue
    candidates_top: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(after_think):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates_top.append(after_think[start:i + 1])
                    start = -1
    for cand in reversed(candidates_top):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    if start != -1 and depth > 0:
        repaired = after_think[start:].rstrip().rstrip("`").rstrip() + ("}" * depth)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    return None


def _render_tool_descriptions(tool_schemas: Sequence[dict]) -> str:
    """Render OpenAI-format tool schemas as plain-text bullet descriptions."""
    lines: list[str] = []
    for ts in tool_schemas:
        fn = ts.get("function") or {}
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = (fn.get("parameters") or {}).get("properties", {}) or {}
        required = set((fn.get("parameters") or {}).get("required", []) or [])
        lines.append(f"- {name}: {desc}")
        for pname, pspec in params.items():
            ptype = pspec.get("type", "any")
            pdesc = pspec.get("description", "")
            penum = pspec.get("enum")
            req = " *(required)*" if pname in required else ""
            extra = f" enum={penum}" if penum else ""
            lines.append(f"    - {pname} ({ptype}){req}{extra}: {pdesc}")
    return "\n".join(lines)


def _example_for_param(pname: str, pspec: dict) -> str:
    """Pick a placeholder for a JSON example (display only — not parsed)."""
    ptype = pspec.get("type", "any")
    if pspec.get("enum"):
        return f'"{pspec["enum"][0]}"'
    if ptype == "string":
        return f'"<{pname}>"'
    if ptype == "number" or ptype == "integer":
        return "<num>"
    if ptype == "boolean":
        return "<true|false>"
    if ptype == "array":
        return "[]"
    if ptype == "object":
        return "{}"
    return f'"<{pname}>"'


def _render_tool_call_examples(tool_schemas: Sequence[dict]) -> str:
    """Render concrete tool-call JSON examples (one per tool) for the prompt."""
    out: list[str] = []
    for ts in tool_schemas:
        fn = ts.get("function") or {}
        name = fn.get("name", "")
        params = (fn.get("parameters") or {}).get("properties", {}) or {}
        required = set((fn.get("parameters") or {}).get("required", []) or [])
        # Show required first, then all others.
        ordered = [k for k in params if k in required] + [k for k in params if k not in required]
        args_inline = ", ".join(
            f'"{k}": {_example_for_param(k, params[k])}' for k in ordered
        )
        out.append(
            f"To call {name}:\n"
            f"```json\n"
            f'{{"thought": "<reasoning>", "{name}": {{{args_inline}}}}}\n'
            f"```"
        )
    return "\n\n".join(out)


# =====================================================================
# Variant V1 — minimal natural-language insert (PRESERVED for record).
# Tool descriptions stay in the original "The tools available:" section;
# we only add a short paragraph telling the model the JSON output shape
# (`{"<tool_name>": {<args>}}`). No JSON schema block.
# =====================================================================
_TOOL_CALL_FORMAT_INSERT_V1 = """\
Format the tool call as a fenced ```json code block with the tool name as the
single key and its arguments dict as the value:
```json
{tool_examples_inline}
```

Do NOT use native tool_call markup (<tool_call>, <|tool_call>, <function=).
Only the fenced ```json block is parsed."""


def _render_tool_examples_inline(tool_schemas: Sequence[dict]) -> str:
    """One-line example per tool, joined as alternatives — kept compact so the
    insertion into the original TURN FORMAT block stays minimal."""
    out: list[str] = []
    for ts in tool_schemas:
        fn = ts.get("function") or {}
        name = fn.get("name", "")
        params = (fn.get("parameters") or {}).get("properties", {}) or {}
        required = set((fn.get("parameters") or {}).get("required", []) or [])
        ordered = [k for k in params if k in required] + [k for k in params if k not in required]
        args_inline = ", ".join(
            f'"{k}": {_example_for_param(k, params[k])}' for k in ordered
        )
        out.append(f'{{"{name}": {{{args_inline}}}}}')
    return "\nor\n".join(out)


def _build_system_prompt_v1_natural(original: str, tool_schemas: Sequence[dict]) -> str:
    """V1 (preserved): minimal natural-language insert of the JSON output
    format. No `<tools>` schema block — the original "The tools available:"
    section in react.py supplies the (informal) tool description.

    Insertion point: between the PLAN/REASONING preamble and the final-answer
    JSON block, identified by the ``Keep each under ~40 words.`` line.
    """
    inline = _render_tool_examples_inline(tool_schemas)
    insert = _TOOL_CALL_FORMAT_INSERT_V1.format(tool_examples_inline=inline)
    anchor = "Keep each under ~40 words. Do not skip this block even when the next action seems obvious."
    if anchor in original:
        return original.replace(anchor, anchor + "\n\n" + insert, 1)
    return original.rstrip() + "\n\n" + insert


# =====================================================================
# Variant V2 — Hermes-style schema block (CURRENT default).
# Mirrors what Qwen3 chat template would auto-inject when tools=[] is sent
# (via OpenAI API), but as plain text so we can keep tools=[] unset and
# bypass vLLM's tool parser. Output asked is fenced ```json with
# {"name":"...","arguments":{...}} — Hermes shape — instead of native
# <tool_call>...</tool_call> markup.
#
# Reference:
#   Qwen3 chat template (tokenizer_config.json) emits exactly this block
#   when `tools` is non-empty. Source: HF Qwen/Qwen3-VL-4B-Thinking.
# =====================================================================
_TOOL_CALL_FORMAT_INSERT_V2 = """\
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_json_lines}
</tools>

For each function call, output a fenced ```json code block with the function
name and arguments dict (Hermes-style — NOT the native <tool_call> markup):
```json
{{"name": "<function-name>", "arguments": <args-json-object>}}
```

Do NOT use native tool_call markup (<tool_call>, <|tool_call>, <function=).
Only the fenced ```json block is parsed."""


def _render_tools_json_lines(tool_schemas: Sequence[dict]) -> str:
    """Each tool's `function` schema (name/description/parameters) on its own
    line, JSON-serialized — same shape that Qwen3 chat template auto-injects
    inside the <tools></tools> block."""
    return "\n".join(
        json.dumps(ts.get("function") or {}, ensure_ascii=False)
        for ts in tool_schemas
    )


def _build_system_prompt(original: str, tool_schemas: Sequence[dict]) -> str:
    """V2 (default): native-equivalent <tools> schema block + Hermes-style
    JSON output instruction. Drop-in replacement for V1 — same insertion
    anchor, same surrounding text. Only the inserted block content differs.

    The model sees the same JSON Schema info as it would if we had passed
    `tools=[]` (and let vLLM's chat template auto-inject), but the output
    instruction asks for a fenced ```json block instead of native
    <tool_call> markup so we bypass vLLM's tool parser entirely.
    """
    schema_block = _render_tools_json_lines(tool_schemas)
    insert = _TOOL_CALL_FORMAT_INSERT_V2.format(tools_json_lines=schema_block)
    anchor = "Keep each under ~40 words. Do not skip this block even when the next action seems obvious."
    if anchor in original:
        return original.replace(anchor, anchor + "\n\n" + insert, 1)
    return original.rstrip() + "\n\n" + insert


class JsonPromptPlanner(OpenAICompatiblePlanner):
    """OpenAI-compatible planner that uses prompt-only JSON for tool calling.

    Designed for Qwen3 / Gemma 4 where native tool_call parsing in vLLM has
    known bugs (#39056, #39468). Preserves the same PlannerLLM interface so
    react.py is unchanged.

    Differences vs OpenAICompatiblePlanner:
      - start_chat: rewrites system_prompt, passes tool_schemas=[] to parent
        (so chat.completions.create won't get tools=/tool_choice=).
      - send: converts ToolCallResult sequence to a single user-role text
        message (no role:tool / tool_call_id).
      - _append_assistant_turn: appends plain content (no tool_calls field).
      - _parse: extracts JSON action from content (or thinking as fallback)
        and synthesizes PlannerToolCall objects with fake call_ids.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_call_counter = 0
        self._tool_names: list[str] = []

    async def start_chat(
        self,
        system_prompt: str,
        tool_schemas: Sequence[dict],
        temperature: float = 0.0,
        thinking_budget: Optional[int] = None,
    ) -> None:
        augmented = _build_system_prompt(system_prompt, tool_schemas)
        # Pass empty tool_schemas so parent stores _tools=[] and
        # _build_request_kwargs won't add tools= / tool_choice=.
        await super().start_chat(augmented, [], temperature, thinking_budget)
        self._tool_call_counter = 0
        self._tool_names = [
            (ts.get("function") or {}).get("name", "") for ts in tool_schemas
        ]
        self._tool_names = [n for n in self._tool_names if n]

    async def send(self, message: PlannerInput) -> Optional[PlannerResponse]:
        if not isinstance(message, str):
            parts: list[str] = []
            for r in message:
                parts.append(f"[Tool '{r.name}' returned]\n{r.result}")
            converted = "\n\n".join(parts)
            converted += (
                "\n\nNow output the next ```json action block "
                '(call another tool, or finalize with the "answer" key).'
            )
            message = converted
        return await super().send(message)

    def _append_assistant_turn(self, resp) -> None:
        """Append plain content to history (no native tool_calls field).
        Falls back to reasoning_content if content is empty (defensive against
        the Qwen Thinking leak case)."""
        msg = resp.choices[0].message
        text = msg.content or ""
        if not text:
            text = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
        self._messages.append({"role": "assistant", "content": text})

    # ----- response → PlannerResponse -------------------------------------

    def _tool_call_response(
        self, name: str, args: dict, text: str, thinking: str
    ) -> PlannerResponse:
        """Pass through model's full text (PLAN/REASONING preamble + the JSON
        block) so react.py can log it. The structured tool_call is what
        actually drives execution."""
        self._tool_call_counter += 1
        synth_id = f"call_jsonprompt_{self._tool_call_counter}"
        return PlannerResponse(
            thinking=thinking,
            text=text,
            tool_calls=[PlannerToolCall(name=name, arguments=args, call_id=synth_id)],
        )

    def _parse(self, resp) -> PlannerResponse:
        msg = resp.choices[0].message
        thinking = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        text = msg.content or ""

        # Try content first; fall back to thinking (covers Qwen Thinking leak
        # where the JSON might end up in reasoning_content).
        parsed = _parse_json_block(text) or _parse_json_block(thinking)
        empty = PlannerResponse(thinking=thinking, text=text, tool_calls=[])

        if not parsed or not isinstance(parsed, dict):
            return empty

        # Final answer: top-level {"answer": "X", "reasoning": "..."} — same
        # shape as the ReAct ORIGINAL prompt asks for. Pass model's text
        # through unchanged so react.py's _extract_answer regex picks it up.
        if "answer" in parsed:
            letter = str(parsed.get("answer", "")).strip().upper()
            if letter and len(letter) == 1 and "A" <= letter <= "Z":
                return PlannerResponse(thinking=thinking, text=text, tool_calls=[])
            return empty

        # V2 (Hermes-shape): {"name": "<tool>", "arguments": {<args>}}
        if "name" in parsed and "arguments" in parsed:
            name = str(parsed.get("name", "")).strip()
            args_v = parsed.get("arguments")
            args = args_v if isinstance(args_v, dict) else {}
            if name and name in self._tool_names:
                return self._tool_call_response(name, args, text, thinking)

        # V1 (tool-as-key): {"<tool_name>": {<args>}} — also tolerated for
        # backward compat with older prompts.
        for key, val in parsed.items():
            if key == "thought":
                continue
            if key in self._tool_names and isinstance(val, dict):
                return self._tool_call_response(key, val, text, thinking)
            break

        return empty
