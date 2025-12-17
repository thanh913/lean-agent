"""Executor for defer-based blueprints: compile, extract tasks, call prover, splice results."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .clients import CompileResult, LeanClient, LLMClient, LLMError


# ---------------------------------------------------------------------------
# Text Utilities (for Lean code extraction)
# ---------------------------------------------------------------------------

_LEAN_FENCE = re.compile(r"```\s*lean4?\s*", re.IGNORECASE)
_GENERIC_FENCE = re.compile(r"```\s*[\w-]*\s*", re.IGNORECASE)


def extract_last_lean_block(text: str | None) -> str | None:
    """Return the content of the final ```lean/lean4 fenced block."""
    if not text:
        return None
    lines = text.splitlines()
    fence_indices = [i for i, ln in enumerate(lines) if _GENERIC_FENCE.match(ln.strip())]
    if len(fence_indices) < 2:
        return None
    for idx in range(len(fence_indices) - 2, -1, -1):
        start, end = fence_indices[idx], fence_indices[idx + 1]
        if start < end and _LEAN_FENCE.match(lines[start].strip()):
            if payload := "\n".join(lines[start + 1 : end]).strip():
                return payload
    start, end = fence_indices[-2], fence_indices[-1]
    return "\n".join(lines[start + 1 : end]).strip() or None


def extract_proof_body(snippet: str | None) -> str:
    """Strip a Lean snippet down to the body following ':= by'."""
    if not snippet:
        return ""
    lines = snippet.splitlines()
    saw_header = False
    body: list[str] = []
    for line in lines:
        if not saw_header:
            if ":= by" in line:
                _, after = line.split(":= by", 1)
                saw_header = True
                if tail := after.strip():
                    body.append(tail)
            continue
        body.append(line)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return ""
    indent = min((len(ln) - len(ln.lstrip()) for ln in body if ln.strip()), default=0)
    if indent:
        body = [ln[indent:] if len(ln) >= indent else ln for ln in body]
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Defer Prelude (Lean 4 tactic)
# ---------------------------------------------------------------------------

DEFER_PRELUDE = """
import Lean
import Mathlib
import Aesop
open Lean Meta Elab Tactic

namespace Defer
/-
defer: Extract proof goals with minimal context for external tools.

Syntax:
  defer h1 h2 ... hn

Outputs JSON with:
  - goal: the target type
  - context: type-closure of goal's free variables + listed identifiers
  - file, start_byte, end_byte, indent: source location info

Type-closure: if x is included, everything needed to state x's type
is also included, recursively. Output is copy-pasteable Lean.

Numeric literals (n : T) are annotated only when T ≠ Nat and no
free variable of type T appears in the expression (grounding principle).
-/

/-- Extract all free variable IDs from an expression. -/
def collectFVarIds (e : Expr) : Array FVarId :=
  (Lean.CollectFVars.main e {}).fvarIds

/-- Compute transitive closure over type dependencies.
    If x is in the set, adds all fvars appearing in x's type (and value, for let-bindings). -/
partial def typeClosure (ctx : LocalContext) (fvars : FVarIdSet) : FVarIdSet :=
  let expanded := Id.run do
    let mut acc := fvars
    for fvarId in fvars do
      match ctx.find? fvarId with
      | none => continue
      | some decl =>
        let fromType := collectFVarIds decl.type
        let fromVal := decl.value?.map collectFVarIds |>.getD #[]
        for fv in fromType ++ fromVal do
          acc := acc.insert fv
    return acc
  if expanded.size > fvars.size then typeClosure ctx expanded else expanded

/-- Check if expression contains a free variable whose type defeq T.
    Used to determine if numeric literals of type T are "grounded".
    Wrapped in withoutModifyingState to avoid unification side effects. -/
def hasGroundingVar (e : Expr) (T : Expr) : MetaM Bool := do
  for fvarId in collectFVarIds e do
    -- Use withoutModifyingState to ensure isDefEq doesn't assign metavariables
    let isMatch ← withoutModifyingState do
      isDefEq (← fvarId.getDecl).type T
    if isMatch then return true
  return false

/-- Traverse expression to check if any numeric literal needs type annotation.
    Returns true iff there exists (OfNat.ofNat T n _) where T ≠ Nat and
    no grounding variable of type T exists in the whole expression. -/
partial def needsNumericAnnotation (whole : Expr) : MetaM Bool := do
  let rec check (e : Expr) : MetaM Bool := do
    -- OfNat.ofNat has 3 arguments: (T : Type) (n : Nat) (inst : OfNat T n)
    if e.isAppOfArity ``OfNat.ofNat 3 then
      let T := e.getAppArgs[0]!
      if !T.isConstOf ``Nat && !(← hasGroundingVar whole T) then
        return true
    match e with
    | .app f a => return (← check f) || (← check a)
    | .lam _ t b _ | .forallE _ t b _ => return (← check t) || (← check b)
    | .letE _ t v b _ => return (← check t) || (← check v) || (← check b)
    | .mdata _ e | .proj _ _ e => check e
    | _ => return false
  check whole

/-- Pretty-print with minimal type annotations.
    Note: If ANY numeric literal requires annotation (ungrounded polymorphic type),
    ALL numeric literals in the expression will be annotated. This is a limitation
    of using pp.numericTypes globally, but ensures correctness. -/
def ppMinimal (e : Expr) : MetaM String := do
  let showTypes ← needsNumericAnnotation e
  withOptions (·.setBool `pp.numericTypes showTypes) do
    return (← ppExpr e).pretty

/-- Format a local declaration as a binder string.
    Handles: (x : T), {x : T}, [T], let x : T := v -/
def formatBinder (decl : LocalDecl) : MetaM String := do
  let type ← ppMinimal decl.type
  if decl.isLet then
    if let some v := decl.value? then
      return s!"let {decl.userName} : {type} := {← ppMinimal v}"
  match decl.binderInfo with
  | .instImplicit => return s!"[{type}]"
  -- Use string concatenation to avoid brace escaping issues
  | .implicit | .strictImplicit =>
      return "{" ++ s!"{decl.userName} : {type}" ++ "}"
  | _ => return s!"({decl.userName} : {type})"

end Defer

/-- Extract goal and context as JSON for external proof tools.
    Lists goal + type-closure of goal's free variables and any explicitly listed hypotheses. -/
elab "defer" args:(colGt ident)* : tactic => do
  withMainContext do
    let ctx ← getLCtx
    let goal ← getMainGoal
    let goalType ← instantiateMVars (← goal.getType)

    -- Seed with goal's free variables + explicitly listed identifiers
    let goalFVars := Defer.collectFVarIds goalType
    let mut seeds : FVarIdSet := goalFVars.foldl (fun acc fv => acc.insert fv) .empty
    for arg in args do
      match ctx.findFromUserName? arg.getId with
      | some decl => seeds := seeds.insert decl.fvarId
      -- Fixed: Use m!"..." for proper interpolation in error messages
      | none => throwError m!"defer: unknown identifier '{arg.getId}'"

    -- Compute type closure
    let closed := Defer.typeClosure ctx seeds

    -- Collect context in declaration order, skip implementation details
    let mut context : Array String := #[]
    for decl in ctx do
      if closed.contains decl.fvarId && !decl.isImplementationDetail then
        context := context.push (← Defer.formatBinder decl)

    -- Source location for replacement
    let stx ← getRef
    let fileMap ← getFileMap
    let startByte := (stx.getPos?.getD 0).byteIdx
    let endByte := (stx.getTailPos?.getD 0).byteIdx

    let output := Json.mkObj [
      ("goal", Json.str (← Defer.ppMinimal goalType)),
      ("context", Json.arr (context.map Json.str)),
      ("file", Json.str (← getFileName)),
      ("start_byte", Json.num startByte),
      ("end_byte", Json.num endByte),
      ("indent", Json.num (fileMap.toPosition ⟨startByte⟩).column)
    ]
    logInfo s!"[DEFER] {output.compress}"
    goal.assign (← mkSorry goalType true)
""".strip()


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class DeferTask:
    """A deferred subgoal extracted from compiler output."""

    goal: str
    context_lines: list[str]
    start_byte: int
    end_byte: int
    indent: int


@dataclass
class TaskSpec:
    """A prepared task ready for the prover."""

    header: str
    prelude_lines: list[str]
    start_byte: int
    end_byte: int
    indent: int


@dataclass
class TaskOutcome:
    """Result from solving a single task."""

    idx: int
    success: bool
    body: str | None
    log: str
    attempts: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class ExecutionResult:
    """Final execution status."""

    success: bool
    stage: str  # "done", "compile", "solve", "verify"
    subgoals: int
    detail: str
    attempts: int
    proof_block: str | None = None
    failed_tasks: list[tuple[int, str]] | None = None
    prover_prompt_tokens: int = 0
    prover_completion_tokens: int = 0
    prover_inference_ms: float = 0.0


@dataclass
class ExecutorConfig:
    """Configuration for the executor."""

    prover_model: str
    prover_sampling: dict[str, Any]
    verify_timeout: int
    max_prover_attempts: int
    session_id: str
    stagger_delay: float = 60.0  # seconds between starting attempts


# ---------------------------------------------------------------------------
# Blueprint Assembly Helpers
# ---------------------------------------------------------------------------


def _dedupe_imports(header_lines: Iterable[str]) -> list[str]:
    """Deduplicate import statements, ensuring 'import Lean' is first."""
    ordered: list[str] = []
    seen: set[str] = set()
    for line in header_lines:
        stripped = line.strip()
        if not stripped or not stripped.startswith("import "):
            continue
        if stripped not in seen:
            ordered.append(stripped)
            seen.add(stripped)
    if "import Lean" not in seen:
        ordered.insert(0, "import Lean")
    return ordered


def _non_import_lines(header_lines: Iterable[str]) -> list[str]:
    """Extract non-import lines from header."""
    return [
        line.rstrip("\n")
        for line in header_lines
        if not line.lstrip().startswith("import ")
    ]


def _build_prefix(imports: list[str], other_lines: list[str]) -> list[str]:
    """Build the full prefix including DEFER_PRELUDE."""
    parts: list[str] = []
    parts.extend(imports)
    if parts:
        parts.append("")
    parts.append(DEFER_PRELUDE)
    parts.append("")
    parts.extend(other_lines)
    if other_lines and other_lines[-1].strip():
        parts.append("")
    return parts


def _verify_prefix(imports: list[str], other_lines: list[str]) -> list[str]:
    """Build prefix for task verification (without DEFER_PRELUDE)."""
    parts: list[str] = []
    parts.extend(imports)
    if parts:
        parts.append("")
    parts.extend(other_lines)
    if other_lines and other_lines[-1].strip():
        parts.append("")
    return parts


def assemble_blueprint(lean_block: str, header_lines: list[str]) -> str:
    """Assemble full Lean code from blueprint and headers."""
    imports = _dedupe_imports(header_lines)
    other_header = _non_import_lines(header_lines)
    prefix_lines = _build_prefix(imports, other_header)
    prefix = "\n".join(prefix_lines).rstrip() + "\n"
    block = lean_block if lean_block.endswith("\n") else f"{lean_block}\n"
    return prefix + block


def get_verify_prefix(header_lines: list[str]) -> list[str]:
    """Get prefix lines for task verification (without DEFER_PRELUDE)."""
    imports = _dedupe_imports(header_lines)
    other_header = _non_import_lines(header_lines)
    return _verify_prefix(imports, other_header)


# ---------------------------------------------------------------------------
# Task Extraction
# ---------------------------------------------------------------------------


def extract_tasks(
    messages: list[dict[str, Any]],
    code: str | None = None,
) -> list[DeferTask]:
    """Extract deferred tasks from compiler messages.

    Args:
        messages: Compiler messages containing [DEFER] info
        code: Original source code (used to compute offset correction
              when Lean's byte positions don't match our source)
    """
    tasks: list[DeferTask] = []
    code_bytes = code.encode("utf-8") if code else None

    # First pass: collect all tasks with raw byte positions
    raw_tasks: list[tuple[dict, dict, int, int, int, list[str]]] = []

    for message in messages or []:
        severity = str(message.get("severity", "")).lower()
        if severity not in {"info", "information"}:
            continue
        data = str(message.get("data", ""))
        if not data.startswith("[DEFER] "):
            continue
        payload_raw = data.replace("[DEFER] ", "", 1)
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue
        context = payload.get("context", [])
        if isinstance(context, list):
            context_lines = [str(line) for line in context]
        else:
            context_lines = str(context).splitlines()

        try:
            start = int(payload.get("start_byte"))
            end = int(payload.get("end_byte"))
        except (TypeError, ValueError):
            continue

        indent = int(payload.get("indent", 0) or 0)
        goal = str(payload.get("goal", ""))
        raw_tasks.append((payload, message, start, end, indent, context_lines))

    if not raw_tasks:
        return tasks

    # Compute offset correction if code is available
    offset = 0
    if code_bytes and raw_tasks:
        # Find where "defer" actually appears and compare to reported position
        # Use first task to compute offset (assumes consistent offset)
        first_start = raw_tasks[0][2]
        first_end = raw_tasks[0][3]
        reported_len = first_end - first_start

        # Search for "defer" near the reported position
        search_start = max(0, first_start - 200)
        search_end = min(len(code_bytes), first_start + 200)
        search_region = code_bytes[search_start:search_end]

        # Find "defer" in search region
        defer_pos = search_region.find(b"defer")
        if defer_pos >= 0:
            actual_start = search_start + defer_pos
            offset = actual_start - first_start

    # Apply offset to all tasks
    for payload, message, start, end, indent, context_lines in raw_tasks:
        tasks.append(
            DeferTask(
                goal=str(payload.get("goal", "")),
                context_lines=context_lines,
                start_byte=start + offset,
                end_byte=end + offset,
                indent=indent,
            )
        )

    return tasks


def _split_context(context_lines: list[str]) -> tuple[list[str], list[str]]:
    """Split context lines into binders and type-level let telescope.

    Strategy:
      - Non-`let` lines are treated as standard binders.
      - For each `let x : T := v`:
          * If `x` appears in any *binder* line, promote to binders
            `(x : T)` and `(h_def_x : x = v)`.
          * Otherwise, keep as type-level let telescope.
    """
    binders: list[str] = []
    let_specs: list[tuple[str, str | None, str | None, str]] = []
    raw_binders: list[str] = []

    for line in context_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("let "):
            rest = stripped[len("let ") :].strip()
            name: str | None = None
            type_str: str | None = None
            value: str | None = None
            if ":=" in rest:
                lhs, rhs = rest.split(":=", 1)
                lhs = lhs.strip()
                value = rhs.strip()
            else:
                lhs = rest
            if ":" in lhs:
                name_part, type_part = lhs.split(":", 1)
                name = name_part.strip()
                type_str = type_part.strip()
            let_specs.append((name or "", type_str, value, stripped))
        else:
            raw_binders.append(stripped)

    # Detect let-bound names referenced in binder types
    binder_text = "\n".join(raw_binders)
    names_used_in_binders: set[str] = set()
    for name, _type_str, _value, _line in let_specs:
        if not name:
            continue
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        if pattern.search(binder_text):
            names_used_in_binders.add(name)

    let_lines_for_type: list[str] = []
    for line in context_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("let "):
            rest = stripped[len("let ") :].strip()
            name = None
            type_str = None
            value = None
            if ":=" in rest:
                lhs, rhs = rest.split(":=", 1)
                lhs = lhs.strip()
                value = rhs.strip()
            else:
                lhs = rest
            if ":" in lhs:
                name_part, type_part = lhs.split(":", 1)
                name = name_part.strip()
                type_str = type_part.strip()
            if name and type_str and name in names_used_in_binders:
                binders.append(f"({name} : {type_str})")
                if value:
                    binders.append(f"(h_def_{name} : {name} = {value})")
            else:
                let_lines_for_type.append(stripped)
        else:
            binders.append(stripped)

    return binders, let_lines_for_type


def prepare_task_specs(tasks: list[DeferTask]) -> list[TaskSpec]:
    """Convert DeferTasks to TaskSpecs ready for the prover."""
    specs: list[TaskSpec] = []
    for idx, task in enumerate(tasks):
        binders, let_lines = _split_context(task.context_lines)
        header_parts = [f"theorem subgoal_{idx}"]
        if binders:
            header_parts.append(" ".join(binders))
        header_prefix = " ".join(header_parts)
        if let_lines:
            type_lines = [f"  {line}" for line in let_lines]
            type_lines.append(f"  {task.goal}")
            header = header_prefix + " :\n" + "\n".join(type_lines) + " := by"
        else:
            header = f"{header_prefix} : {task.goal} := by"
        specs.append(
            TaskSpec(
                header=header,
                prelude_lines=[],
                start_byte=task.start_byte,
                end_byte=task.end_byte,
                indent=task.indent,
            )
        )
    return specs


def render_task_snippet(spec: TaskSpec, verify_prefix: list[str]) -> str:
    """Render a task as a complete Lean snippet with sorry."""
    snippet_lines: list[str] = [*verify_prefix, spec.header]
    for line in spec.prelude_lines:
        snippet_lines.append(f"  {line}")
    snippet_lines.append("  sorry")
    return "\n".join(snippet_lines) + "\n"


def render_subgoal(spec: TaskSpec) -> str:
    """Render a task spec as a displayable subgoal."""
    lines: list[str] = [spec.header]
    lines.extend(f"  {line}" if line else "" for line in spec.prelude_lines)
    lines.append("  sorry")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Proof Splicing
# ---------------------------------------------------------------------------


def extract_clean_proof(spliced_code: str, header_lines: list[str]) -> str:
    """Extract clean proof from spliced code (without DEFER_PRELUDE).

    Args:
        spliced_code: Code with proofs spliced in (includes DEFER_PRELUDE)
        header_lines: Original header lines

    Returns:
        Clean code with just imports and theorem (no Defer namespace)
    """
    lines = spliced_code.splitlines()

    # The DEFER_PRELUDE ends with the elab block containing "goal.assign"
    # Find that line and look for the first theorem/lemma after it
    elab_end_idx = None
    for i, line in enumerate(lines):
        if "goal.assign" in line:
            elab_end_idx = i
            break

    # Start searching for theorem after the elab block ends
    search_start = elab_end_idx + 1 if elab_end_idx is not None else 0

    # Find where the theorem starts (first line starting with theorem, lemma, etc.)
    # These are the standard Lean declaration keywords
    decl_keywords = ("theorem ", "lemma ", "example ", "corollary ", "proposition ")
    theorem_start_idx = None
    for i in range(search_start, len(lines)):
        stripped = lines[i].lstrip()
        if any(stripped.startswith(kw) for kw in decl_keywords):
            theorem_start_idx = i
            break

    if theorem_start_idx is None:
        # No theorem found, return as-is
        return spliced_code

    # Get the theorem and everything after it
    theorem_lines = lines[theorem_start_idx:]

    # Build clean prefix (imports without DEFER_PRELUDE)
    verify_prefix = get_verify_prefix(header_lines)

    # Combine
    result_lines = verify_prefix + theorem_lines
    return "\n".join(result_lines) + "\n" if result_lines else ""


def splice_proofs(
    code: str,
    tasks: list[DeferTask],
    outcomes: list[TaskOutcome],
) -> str:
    """Splice proven bodies back into code, replacing defer calls.

    Args:
        code: Original code containing defer calls
        tasks: List of DeferTask with byte positions (UTF-8)
        outcomes: List of TaskOutcome with proof bodies (must match tasks by idx)

    Returns:
        Code with defer calls replaced by proof bodies
    """
    if not tasks or not outcomes:
        return code

    # Build outcome map by idx
    outcome_map = {o.idx: o for o in outcomes if o.success and o.body}

    # Sort tasks by start_byte descending (splice from end to preserve positions)
    indexed_tasks = [(i, t) for i, t in enumerate(tasks)]
    indexed_tasks.sort(key=lambda x: x[1].start_byte, reverse=True)

    # Work with bytes since Lean reports UTF-8 byte positions
    result_bytes = code.encode("utf-8")

    for idx, task in indexed_tasks:
        outcome = outcome_map.get(idx)
        if not outcome or not outcome.body:
            continue

        # Format body with proper indentation
        body_lines = outcome.body.strip().splitlines()
        if len(body_lines) == 1:
            # Single line: just replace inline
            indented_body = body_lines[0]
        else:
            # Multi-line: indent continuation lines
            indent = " " * task.indent
            indented_body = body_lines[0]
            for line in body_lines[1:]:
                if line.strip():
                    indented_body += "\n" + indent + line
                else:
                    indented_body += "\n"

        # Replace the defer call with the proof body (using byte positions)
        body_bytes = indented_body.encode("utf-8")
        result_bytes = result_bytes[:task.start_byte] + body_bytes + result_bytes[task.end_byte:]

    return result_bytes.decode("utf-8")


# ---------------------------------------------------------------------------
# Prover Prompt
# ---------------------------------------------------------------------------

PROVER_PROMPT_TEMPLATE = """\
Complete the following Lean 4 code:

```lean4
{snippet}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan \
outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the \
construction of the final formal proof."""


def build_prover_prompt(snippet: str) -> str:
    """Build the prover prompt for a snippet."""
    return PROVER_PROMPT_TEMPLATE.format(snippet=snippet.strip())


# ---------------------------------------------------------------------------
# Proof Executor
# ---------------------------------------------------------------------------


class ProofExecutor:
    """Executes defer-based blueprints by solving subgoals with an LLM prover."""

    def __init__(
        self,
        llm: LLMClient,
        lean: LeanClient,
        config: ExecutorConfig,
    ) -> None:
        self._llm = llm
        self._lean = lean
        self._config = config

    async def execute(
        self,
        blueprint: str,
        header_lines: list[str],
    ) -> ExecutionResult:
        """Execute a blueprint: compile, extract tasks, solve, verify."""
        # 1. Compile blueprint with sorries allowed
        raw_blueprint = extract_last_lean_block(blueprint) or blueprint
        code_with_prelude = assemble_blueprint(raw_blueprint, header_lines)

        compile_result = await self._lean.compile(
            code=code_with_prelude,
            timeout=self._config.verify_timeout,
            allow_sorry=True,
            snippet_id=f"{self._config.session_id}-blueprint",
        )

        if not compile_result.ok:
            detail = "Plan failed to compile."
            if compile_result.error_log:
                detail = f"{detail}\n```lean4\n{compile_result.error_log}\n```"
            return ExecutionResult(
                success=False,
                stage="compile",
                subgoals=0,
                detail=detail,
                attempts=0,
            )

        # 2. Extract tasks (pass code for reliable byte position computation)
        tasks = extract_tasks(compile_result.messages, code_with_prelude)
        if not tasks:
            return await self._final_verify_only(code_with_prelude)

        # 3. Prepare task specs
        task_specs = prepare_task_specs(tasks)
        verify_prefix = get_verify_prefix(header_lines)

        # 4. Solve all tasks in parallel
        #    LLMClient's semaphore handles concurrency - simple gather!
        outcomes = await asyncio.gather(
            *[
                self._solve_task(idx, spec, verify_prefix)
                for idx, spec in enumerate(task_specs)
            ]
        )

        # 5. Aggregate stats
        attempts_total = sum(o.attempts for o in outcomes)
        prover_prompt_tokens = sum(o.prompt_tokens for o in outcomes)
        prover_completion_tokens = sum(o.completion_tokens for o in outcomes)
        prover_inference_ms = sum(o.latency_ms for o in outcomes)

        rendered_subgoals = "\n\n".join(render_subgoal(spec) for spec in task_specs)
        proof_block = (
            f"/- Subgoals sent to the Prover -/\n{rendered_subgoals}\n"
            if rendered_subgoals
            else None
        )

        # 6. Check for failures
        failures = [(o.idx, o.log) for o in outcomes if not o.success]
        if failures:
            detail_lines = ["Executor failed to solve some subgoals."]
            detail_lines.extend(f"subgoal {idx} failed" for idx, _ in failures)
            return ExecutionResult(
                success=False,
                stage="solve",
                subgoals=len(task_specs),
                detail="\n".join(detail_lines),
                attempts=attempts_total,
                proof_block=proof_block,
                failed_tasks=failures,
                prover_prompt_tokens=prover_prompt_tokens,
                prover_completion_tokens=prover_completion_tokens,
                prover_inference_ms=prover_inference_ms,
            )

        # 7. All tasks succeeded - splice proofs back and final verify
        spliced_code = splice_proofs(code_with_prelude, tasks, outcomes)

        final_check = await self._lean.compile(
            code=spliced_code,
            timeout=self._config.verify_timeout,
            allow_sorry=False,
            snippet_id=f"{self._config.session_id}-final",
        )

        if not final_check.ok:
            detail = "All subgoals solved but final verification failed."
            if final_check.error_log:
                detail = f"{detail}\n```lean4\n{final_check.error_log}\n```"
            return ExecutionResult(
                success=False,
                stage="verify",
                subgoals=len(task_specs),
                detail=detail,
                attempts=attempts_total,
                proof_block=spliced_code,
                prover_prompt_tokens=prover_prompt_tokens,
                prover_completion_tokens=prover_completion_tokens,
                prover_inference_ms=prover_inference_ms,
            )

        # Extract clean proof (without DEFER_PRELUDE) for the final result
        clean_proof = extract_clean_proof(spliced_code, header_lines)

        return ExecutionResult(
            success=True,
            stage="done",
            subgoals=len(task_specs),
            detail=f"Executor solved all deferred subgoals. ({len(task_specs)} subgoal(s))",
            attempts=attempts_total,
            proof_block=clean_proof,
            prover_prompt_tokens=prover_prompt_tokens,
            prover_completion_tokens=prover_completion_tokens,
            prover_inference_ms=prover_inference_ms,
        )

    async def _final_verify_only(self, code_with_prelude: str) -> ExecutionResult:
        """Handle case where blueprint has no deferred tasks."""
        final_check = await self._lean.compile(
            code=code_with_prelude,
            timeout=self._config.verify_timeout,
            allow_sorry=False,
            snippet_id=f"{self._config.session_id}-final",
        )
        if not final_check.ok:
            detail = "Blueprint compiled but final verification failed."
            if final_check.error_log:
                detail = f"{detail}\n```lean4\n{final_check.error_log}\n```"
            return ExecutionResult(
                success=False,
                stage="verify",
                subgoals=0,
                detail=detail,
                attempts=0,
            )
        return ExecutionResult(
            success=True,
            stage="done",
            subgoals=0,
            detail="Executor succeeded with no deferred subgoals.",
            attempts=0,
        )

    async def _solve_task(
        self,
        idx: int,
        spec: TaskSpec,
        verify_prefix: list[str],
    ) -> TaskOutcome:
        """Solve a single task with staggered attempts.

        Only spawns the next attempt after the previous one has actually
        started executing (acquired the semaphore) AND stagger_delay has
        elapsed since that start. This prevents queueing all attempts at
        the semaphore under congestion.
        """
        snippet = render_task_snippet(spec, verify_prefix)
        prompt = build_prover_prompt(snippet)
        stagger_delay = self._config.stagger_delay
        loop = asyncio.get_running_loop()

        pending: set[asyncio.Task] = set()
        started_events: list[asyncio.Event] = []
        next_attempt_idx = 0
        watched_attempt_idx = -1  # Which attempt we're waiting to start
        start_time_of_watched: float | None = None  # When watched attempt started

        completed_attempts = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency_ms = 0.0
        last_log = ""

        def _process_done(done_tasks: set[asyncio.Task]) -> TaskOutcome | None:
            """Process completed tasks, return TaskOutcome if success found."""
            nonlocal completed_attempts, total_prompt_tokens, total_completion_tokens
            nonlocal total_latency_ms, last_log

            for task in done_tasks:
                completed_attempts += 1
                try:
                    result = task.result()
                except Exception as e:
                    last_log = str(e)
                    continue

                # Accumulate metrics from completed attempt
                total_prompt_tokens += result.get("prompt_tokens", 0)
                total_completion_tokens += result.get("completion_tokens", 0)
                total_latency_ms += result.get("latency_ms", 0.0)

                if result.get("success"):
                    return TaskOutcome(
                        idx=idx,
                        success=True,
                        body=result["body"],
                        log="",
                        attempts=completed_attempts,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        latency_ms=total_latency_ms,
                    )
                else:
                    last_log = result.get("log", "Unknown error")
            return None

        def _spawn_attempt() -> None:
            """Spawn the next attempt with a started_event."""
            nonlocal next_attempt_idx, watched_attempt_idx, start_time_of_watched
            event = asyncio.Event()
            started_events.append(event)
            task = asyncio.create_task(
                self._single_attempt(
                    idx, next_attempt_idx, spec, verify_prefix, prompt,
                    started_event=event,
                ),
                name=f"task{idx}-attempt{next_attempt_idx}",
            )
            pending.add(task)
            watched_attempt_idx = next_attempt_idx
            start_time_of_watched = None  # Will be set when event fires
            next_attempt_idx += 1

        def _check_watched_started() -> bool:
            """Check if watched attempt has started, update timing if so."""
            nonlocal start_time_of_watched
            if watched_attempt_idx < 0 or watched_attempt_idx >= len(started_events):
                return False
            event = started_events[watched_attempt_idx]
            if event.is_set() and start_time_of_watched is None:
                start_time_of_watched = loop.time()
                return True
            return start_time_of_watched is not None

        def _can_spawn_next() -> bool:
            """Check if we can spawn the next attempt."""
            if next_attempt_idx >= self._config.max_prover_attempts:
                return False
            if next_attempt_idx == 0:
                return True  # Always spawn first immediately
            # Need watched attempt to have started + stagger_delay elapsed
            if start_time_of_watched is None:
                return False
            elapsed = loop.time() - start_time_of_watched
            return elapsed >= stagger_delay

        # Main loop
        while next_attempt_idx < self._config.max_prover_attempts or pending:
            # Spawn next attempt if conditions are met
            if _can_spawn_next():
                _spawn_attempt()

            if not pending:
                break

            # Determine timeout for wait
            timeout: float | None = None
            if next_attempt_idx < self._config.max_prover_attempts:
                if start_time_of_watched is not None:
                    # Waiting for stagger_delay to elapse
                    elapsed = loop.time() - start_time_of_watched
                    timeout = max(0.01, stagger_delay - elapsed)
                else:
                    # Waiting for watched attempt to start - poll periodically
                    timeout = 0.1

            # Wait for task completion or timeout
            try:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except Exception:
                break

            # Check if watched attempt has started
            _check_watched_started()

            # Process completed tasks
            if done:
                outcome = _process_done(done)
                if outcome:
                    # Cancel remaining pending attempts
                    for p in pending:
                        p.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    return outcome

        # All attempts failed
        return TaskOutcome(
            idx=idx,
            success=False,
            body=None,
            log=last_log,
            attempts=completed_attempts,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            latency_ms=total_latency_ms,
        )

    async def _single_attempt(
        self,
        task_idx: int,
        attempt: int,
        spec: TaskSpec,
        verify_prefix: list[str],
        prompt: str,
        started_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        """Execute a single proof attempt. Returns dict with result info.

        Args:
            task_idx: Index of the task being solved
            attempt: Attempt number (0-indexed)
            spec: Task specification
            verify_prefix: Code prefix for verification
            prompt: The prover prompt
            started_event: Optional event to set when LLM call actually starts
                          (after acquiring semaphore)
        """
        result: dict[str, Any] = {
            "success": False,
            "body": None,
            "log": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0.0,
        }

        # Call LLM
        try:
            response = await self._llm.call(
                model=self._config.prover_model,
                messages=[{"role": "user", "content": prompt}],
                request_id=f"{self._config.session_id}-task{task_idx}-attempt{attempt}",
                started_event=started_event,
                **self._config.prover_sampling,
            )
        except LLMError as e:
            result["log"] = str(e)
            return result

        result["prompt_tokens"] = response.prompt_tokens
        result["completion_tokens"] = response.completion_tokens
        result["latency_ms"] = response.latency_ms

        # Extract proof body
        lean_block = extract_last_lean_block(response.content)
        if not lean_block:
            result["log"] = "No Lean code block in response"
            return result

        body = extract_proof_body(lean_block)
        if not body.strip():
            result["log"] = "No proof body extracted"
            return result

        # Verify the proof
        body_lines = [
            f"  {line}" if line else "" for line in spec.prelude_lines
        ]
        body_lines.extend(
            f"  {line}" if line else "" for line in body.strip().splitlines()
        )
        verify_code = (
            "\n".join([*verify_prefix, spec.header, *body_lines]) + "\n"
        )

        verify_result = await self._lean.compile(
            code=verify_code,
            timeout=self._config.verify_timeout,
            allow_sorry=False,
            snippet_id=f"{self._config.session_id}-task{task_idx}-verify{attempt}",
        )

        if verify_result.ok:
            merged_lines = spec.prelude_lines + body.strip().splitlines()
            result["success"] = True
            result["body"] = "\n".join(merged_lines)
        else:
            result["log"] = verify_result.error_log or "Verification failed"

        return result
