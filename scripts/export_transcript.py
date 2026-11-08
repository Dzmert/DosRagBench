"""Export a Claude Code session transcript (JSONL) to a readable Markdown file.

Keeps the human turns and the assistant's prose; compresses tool calls to a
one-line note and drops the (large) tool outputs, so the result is small enough
to paste or upload into a claude.ai chat.

Usage:
    python scripts/export_transcript.py <session.jsonl> [-o out.md] [--full]
    --full  also include a truncated preview of tool inputs/results
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _text_blocks(content) -> list[str]:
    """Pull human-readable text out of an API message `content` field."""
    if isinstance(content, str):
        return [content]
    out = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            out.append(block["text"])
        elif btype == "tool_use":
            name = block.get("name", "tool")
            inp = block.get("input", {})
            hint = inp.get("description") or inp.get("command") or inp.get("file_path") or ""
            hint = re.sub(r"\s+", " ", str(hint))[:100]
            out.append(f"_[tool: {name}{(' — ' + hint) if hint else ''}]_")
    return out


def _is_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    # Slash-command scaffolding and injected reminders the user never typed.
    if t.startswith("<local-command") or t.startswith("<command-"):
        return True
    if t.startswith("Caveat:") or t.startswith("<system-reminder>"):
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path, help="Path to the session .jsonl")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--full", action="store_true", help="Include tool call notes (default already compact)")
    args = ap.parse_args()

    out = args.out or args.session.with_suffix(".md")
    lines: list[str] = ["# Conversation transcript", ""]

    for raw in args.session.open():
        try:
            o = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if o.get("type") not in ("user", "assistant"):
            continue
        if o.get("isMeta"):
            continue
        msg = o.get("message", {})
        role = msg.get("role", o["type"])
        blocks = _text_blocks(msg.get("content"))
        # Skip tool_result-only user turns and injected noise.
        blocks = [b for b in blocks if not _is_noise(b)]
        if not args.full:
            blocks = [b for b in blocks if not b.startswith("_[tool:")]
        text = "\n\n".join(blocks).strip()
        if not text:
            continue
        speaker = "## User" if role == "user" else "## Assistant"
        lines += [speaker, "", text, ""]

    out.write_text("\n".join(lines))
    kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({kb:.0f} KB, {len(lines)} lines)")


if __name__ == "__main__":
    main()
