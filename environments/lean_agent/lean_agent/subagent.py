"""Subagent for isolated exploration with compiler access."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from .lean_utils import format_check_result, lean_check
from .prompts import format_subagent_instruction
from .xml_parser import parse_lean_check


async def run_subagent(
    parent_messages: list[dict[str, Any]],
    task: str,
    client: AsyncOpenAI,
    model: str,
    verification_url: str,
    verification_key: str,
    verify_timeout: int,
    max_turns: int = 5,
) -> str:
    """Run subagent as a clone with parent's full context."""
    messages: list[dict[str, Any]] = [
        *parent_messages,
        {"role": "user", "content": format_subagent_instruction(task)},
    ]

    content = ""
    for turn in range(max_turns):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stop=["</lean_check>"],
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            return f"Clone error: {e}"

        messages.append({"role": "assistant", "content": content})

        # No tool call = clone is done
        code = parse_lean_check(content)
        if not code:
            return content

        # Tool called - verify and continue
        result = await lean_check(
            code=code,
            verification_url=verification_url,
            verification_key=verification_key,
            timeout=verify_timeout,
            snippet_id=f"clone-{turn}",
        )
        messages.append({"role": "user", "content": format_check_result(result)})

    # Turn limit reached - get final summary
    messages.append({"role": "user", "content": "Turn limit reached. Summarize what you found and include any working code."})
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content or content
    except Exception:
        return content
