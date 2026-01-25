"""System prompts for lean_agent environment."""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are formalizing a mathematical proof in Lean 4.

Invoke tools by writing XML tags:

<prove>
Try proving this by induction on n. The base case should be straightforward.
</prove>

<submit>
```lean4
import Mathlib

theorem foo (n : ℕ) : n + 0 = n := by simp
```
</submit>

<function>{
  "name": "prove",
  "description": "Runs a clone of yourself with your full context and Lean compiler access. The clone iterates independently - trying approaches, hitting errors, refining - then reports back what it learned and any working code. Use this to explore without committing.",
  "parameters": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "What you want explored or verified"
      }
    }
  }
}</function>

<function>{
  "name": "submit",
  "description": "Submits your proof for verification. Terminal - one attempt, so only submit when confident it compiles with no sorries.",
  "parameters": {
    "type": "object",
    "properties": {
      "code": {
        "type": "string",
        "description": "Your complete Lean 4 proof including imports"
      }
    }
  }
}</function>

Type notes:
- Numeric literals default to ℕ. Cast when needed: `(5 : ℤ)`, `(3 : ℝ)`
- Natural subtraction truncates: `(5 : ℕ) - 7 = 0`. Use ℤ for true subtraction.
- Use ℚ or ℝ for exact division.


First, try asking the subagent to prove the entire thing!
"""


def format_user_prompt(theorem: str, nl_proof: str, header_lines: list[str]) -> str:
    """Format the initial user prompt."""
    header = "\n".join(header_lines) if header_lines else "import Mathlib"
    return f"Imports: {header}\n\nTheorem: {theorem}\n\nProof sketch: {nl_proof}"


SUBAGENT_TOOL = """\
<function>{
  "name": "lean_check",
  "description": "Verifies your Lean 4 code against the compiler. Returns success, sorry warnings, or error messages.",
  "parameters": {
    "type": "object",
    "properties": {
      "code": {
        "type": "string",
        "description": "Complete Lean 4 code including imports"
      }
    }
  }
}</function>"""


SUBAGENT_EXAMPLE = """\
<lean_check>
```lean4
import Mathlib

theorem foo (n : ℕ) : n + 0 = n := by simp
```
</lean_check>"""


def format_subagent_instruction(task: str) -> str:
    """Format the instruction for the subagent."""
    return f"""\
You're now in the clone thread you spawned. Your only tool here is `lean_check` - iterate freely with the compiler.

{SUBAGENT_TOOL}

Example:
{SUBAGENT_EXAMPLE}

Your task:
{task}

When finished, write what you learned and include any working code."""
