"""LeanAgentEnv - Multi-turn environment for Lean proof formalization."""

from __future__ import annotations

import os
from typing import Any

import verifiers as vf
from verifiers.types import Messages, State

from .lean_utils import format_check_result, lean_check
from .subagent import run_subagent
from .xml_parser import parse_prove, parse_submit


def _fix_truncated_tag(content: str, tag: str) -> str:
    """Append closing tag if truncated by stop sequence."""
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if open_tag in content and close_tag not in content:
        return content + close_tag
    return content


class LeanAgentEnv(vf.MultiTurnEnv):
    """Multi-turn environment for Lean proof formalization with subagent decomposition."""

    def __init__(
        self,
        *,
        verification_url: str = "",
        verification_api_key_env: str = "VERIFICATION_KEY",
        verify_timeout: int = 60,
        subagent_max_turns: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.verification_url = verification_url
        self.verification_api_key_env = verification_api_key_env
        self.verify_timeout = verify_timeout
        self.subagent_max_turns = subagent_max_turns

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        info = state.get("info") or {}
        state["verification_url"] = info.get("verification_url", self.verification_url) or ""
        state["verification_key"] = os.getenv(self.verification_api_key_env, "")
        state["verified"] = False
        state["submitted"] = False
        state["subagent_calls"] = 0
        return state

    @vf.stop
    async def proof_submitted(self, state: State) -> bool:
        return state.get("submitted", False)

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> Messages:
        # Get last assistant content
        last_msg = messages[-1] if messages else {}
        content = last_msg.get("content", "") if isinstance(last_msg, dict) else ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))

        if not content:
            return [{"role": "user", "content": "No response. Use <prove> or <submit>."}]

        # Fix truncated tags in trajectory (stop sequence cuts off closing tag)
        if "<submit>" in content or "<prove>" in content:
            fixed = _fix_truncated_tag(content, "submit")
            fixed = _fix_truncated_tag(fixed, "prove")
            if fixed != content and isinstance(last_msg, dict):
                last_msg["content"] = fixed

        # Try <submit> - terminal action
        code = parse_submit(content)
        if code:
            state["submitted"] = True
            result = await lean_check(
                code=code,
                verification_url=state["verification_url"],
                verification_key=state["verification_key"],
                timeout=self.verify_timeout,
                snippet_id=f"submit-{len(state.get('trajectory', []))}",
            )
            state["verified"] = result.ok and not result.has_sorry
            return [{"role": "user", "content": format_check_result(result)}]

        # Try <prove>
        task = parse_prove(content)
        if task:
            state["subagent_calls"] = state.get("subagent_calls", 0) + 1
            result = await run_subagent(
                parent_messages=messages,
                task=task,
                client=state["client"],
                model=state["model"],
                verification_url=state["verification_url"],
                verification_key=state["verification_key"],
                verify_timeout=self.verify_timeout,
                max_turns=self.subagent_max_turns,
            )
            return [{"role": "user", "content": f"<agent_result>\n{result}\n</agent_result>"}]

        # No valid tag
        return [{"role": "user", "content": "Use <prove> or <submit> to continue."}]
