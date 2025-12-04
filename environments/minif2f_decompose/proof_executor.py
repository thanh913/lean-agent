"""Executor for defer-based blueprints: compile, extract tasks, call prover, splice results."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .lean_utils import extract_last_lean_block, lean_compile
from .prover import Prover, ProverRequest


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


@dataclass
class DeferTask:
    goal: str
    context_lines: list[str]
    start_byte: int
    end_byte: int
    indent: int


@dataclass
class TaskSpec:
    header: str
    prelude_lines: list[str]
    start_byte: int
    end_byte: int
    indent: int


@dataclass
class TaskOutcome:
    idx: int
    attempts: int
    success: bool
    body: str | None
    log: str
    start_byte: int
    end_byte: int
    indent: int
    prover_prompt_tokens: int = 0
    prover_completion_tokens: int = 0
    prover_inference_ms: float = 0.0


@dataclass
class ExecutionResult:
    success: bool
    stage: str
    subgoals: int
    detail: str
    attempts: int
    proof_block: str | None = None
    failed_tasks: list[tuple[int, str]] | None = None
    prover_prompt_tokens: int = 0
    prover_completion_tokens: int = 0
    prover_inference_ms: float = 0.0


def _dedupe_imports(header_lines: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for line in header_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("import "):
            continue
        if stripped not in seen:
            ordered.append(stripped)
            seen.add(stripped)
    if "import Lean" not in seen:
        ordered.insert(0, "import Lean")
    return ordered


def _non_import_lines(header_lines: Iterable[str]) -> list[str]:
    return [line.rstrip("\n") for line in header_lines if not line.lstrip().startswith("import ")]


def _build_prefix(imports: list[str], other_lines: list[str]) -> list[str]:
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


def _common_prefix(imports: list[str], other_lines: list[str]) -> list[str]:
    parts: list[str] = []
    parts.extend(imports)
    if parts:
        parts.append("")
    parts.extend(other_lines)
    if other_lines and other_lines[-1].strip():
        parts.append("")
    return parts


def _assemble_blueprint(lean_block: str, header_lines: list[str]) -> tuple[str, list[str], list[str]]:
    imports = _dedupe_imports(header_lines)
    other_header = _non_import_lines(header_lines)
    prefix_lines = _build_prefix(imports, other_header)
    prefix = "\n".join(prefix_lines).rstrip() + "\n"
    block = lean_block if lean_block.endswith("\n") else f"{lean_block}\n"
    return prefix + block, imports, other_header


def _render_subgoal(spec: TaskSpec) -> str:
    snippet_lines: list[str] = [spec.header]
    snippet_lines.extend(f"  {line}" if line else "" for line in spec.prelude_lines)
    snippet_lines.append("  sorry")
    return "\n".join(snippet_lines)


def _extract_tasks(messages: list[dict]) -> list[DeferTask]:
    tasks: list[DeferTask] = []
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
        tasks.append(
            DeferTask(
                goal=str(payload.get("goal", "")),
                context_lines=context_lines,
                start_byte=start,
                end_byte=end,
                indent=indent,
            )
        )
    return tasks


def _split_context(context_lines: list[str]) -> tuple[list[str], list[str]]:
    """Split context lines into binders and optional type-level `let` telescope.

    Strategy:
      - Non-`let` lines are treated as standard binders and kept in order.
      - For each `let x : T := v`:
          * If `x` appears in any *binder* line (e.g. `h : ... x ...`), we promote
            it to binders `(x : T)` and `(h_def_x : x = v)` so that binder types
            can refer to `x` safely.
          * Otherwise, we keep the `let` as part of a type-level telescope that
            will be inserted after the colon and before the goal:

                theorem subgoal (a b c : ℝ) (h : ...) :
                  let x : ℝ := ...
                  let y : ℝ := ...
                  goal := by

    The returned pair is:
      - binders: list of binder strings to appear before the colon.
      - let_lines: list of `let` lines to appear in the theorem type telescope.
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

    # Detect let-bound names that are referenced inside binder types.
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
            # Re-parse to avoid depending on earlier loop state.
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
            if name and type_str and name in names_used_in_binders:
                # Promote to binders: (x : T) (h_def_x : x = v)
                binders.append(f"({name} : {type_str})")
                if value:
                    binders.append(f"(h_def_{name} : {name} = {value})")
            else:
                # Safe to keep as a type-level let.
                let_lines_for_type.append(stripped)
        else:
            binders.append(stripped)

    return binders, let_lines_for_type


async def _prove_single_task(
    *,
    idx: int,
    task: TaskSpec,
    common_prefix: list[str],
    prover: Prover,
    backend_url: str,
    verify_timeout: int,
    max_prover_attempts: int,
    session_id: str,
) -> TaskOutcome:
    snippet_lines: list[str] = [*common_prefix, task.header]
    for line in task.prelude_lines:
        snippet_lines.append(f"  {line}")
    snippet_lines.append("  sorry")
    snippet = "\n".join(snippet_lines) + "\n"

    total_attempts = 0
    last_log = ""
    body: str | None = None
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_inference_ms = 0.0
    attempt_budget = max(1, max_prover_attempts)
    for attempt in range(attempt_budget):
        request = ProverRequest(
            theorem_snippet=snippet,
            session_id=f"{session_id}-task{idx}-attempt{attempt}",
        )
        result = await prover.prove(request, max_attempts=1)
        total_attempts += result.attempts
        total_prompt_tokens += getattr(result, "prompt_tokens", 0) or 0
        total_completion_tokens += getattr(result, "completion_tokens", 0) or 0
        total_inference_ms += float(getattr(result, "inference_ms", 0.0) or 0.0)
        candidate_body = (result.body or "").strip()
        if not result.success or not candidate_body:
            last_log = result.log
            continue
        body_lines = [f"  {line}" if line else "" for line in task.prelude_lines]
        body_lines.extend(f"  {line}" if line else "" for line in candidate_body.splitlines())
        candidate_code = "\n".join([*common_prefix, task.header, *body_lines]) + "\n"
        compile_result = await lean_compile(
            code=candidate_code,
            backend_url=backend_url,
            timeout=verify_timeout,
            allow_sorry=False,
            snippet_id=f"{session_id}-task{idx}-attempt{attempt}",
        )
        if compile_result.ok:
            merged_lines = task.prelude_lines + candidate_body.splitlines()
            body = "\n".join(merged_lines)
            last_log = ""
            break
        last_log = compile_result.log
    return TaskOutcome(
        idx=idx,
        attempts=total_attempts,
        success=body is not None,
        body=body,
        log=last_log,
        start_byte=task.start_byte,
        end_byte=task.end_byte,
        indent=task.indent,
        prover_prompt_tokens=total_prompt_tokens,
        prover_completion_tokens=total_completion_tokens,
        prover_inference_ms=total_inference_ms,
    )


async def execute_plan(
    *,
    plan_text: str,
    header_lines: list[str],
    prover: Prover,
    backend_url: str,
    verify_timeout: int,
    max_prover_attempts: int,
    session_id: str,
) -> ExecutionResult:
    """Compile the planner blueprint, solve its subgoals, and track prover stats."""

    lean_block = extract_last_lean_block(plan_text) or plan_text
    full_code, imports, other_header = _assemble_blueprint(lean_block, header_lines)

    compile_result = await lean_compile(
        code=full_code,
        backend_url=backend_url,
        timeout=verify_timeout,
        allow_sorry=True,
        snippet_id=f"{session_id}-blueprint",
    )
    if not compile_result.ok:
        detail = "Plan failed to compile."
        if compile_result.log:
            detail = f"{detail}\n```lean4\n{compile_result.log}\n```"
        return ExecutionResult(
            success=False,
            stage="plan_compile",
            subgoals=0,
            detail=detail,
            attempts=0,
        )

    tasks = _extract_tasks(compile_result.messages)
    if not tasks:
        final_check = await lean_compile(
            code=full_code,
            backend_url=backend_url,
            timeout=verify_timeout,
            allow_sorry=False,
            snippet_id=f"{session_id}-final",
        )
        if not final_check.ok:
            detail = "Blueprint compiled but final verification failed."
            if final_check.log:
                detail = f"{detail}\n```lean4\n{final_check.log}\n```"
            return ExecutionResult(
                success=False,
                stage="final_verify",
                subgoals=0,
                detail=detail,
                attempts=0,
            )
        return ExecutionResult(
            success=True,
            stage="success",
            subgoals=0,
            detail="Executor succeeded with no deferred subgoals.",
            attempts=0,
        )

    common_prefix = _common_prefix(imports, other_header)
    task_specs: list[TaskSpec] = []
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
        task_specs.append(
            TaskSpec(
                header=header,
                prelude_lines=[],
                start_byte=task.start_byte,
                end_byte=task.end_byte,
                indent=task.indent,
            )
        )

    jobs = [
        asyncio.create_task(
            _prove_single_task(
                idx=idx,
                task=spec,
                common_prefix=common_prefix,
                prover=prover,
                backend_url=backend_url,
                verify_timeout=verify_timeout,
                max_prover_attempts=max_prover_attempts,
                session_id=session_id,
            )
        )
        for idx, spec in enumerate(task_specs)
    ]
    outcomes = await asyncio.gather(*jobs)

    attempts_total = sum(outcome.attempts for outcome in outcomes)
    failures = [(outcome.idx, outcome.log) for outcome in outcomes if not outcome.success]
    prover_prompt_tokens = sum(outcome.prover_prompt_tokens for outcome in outcomes)
    prover_completion_tokens = sum(outcome.prover_completion_tokens for outcome in outcomes)
    prover_inference_ms = sum(outcome.prover_inference_ms for outcome in outcomes)
    rendered_subgoals = "\n\n".join(_render_subgoal(spec) for spec in task_specs)
    proof_block = ""
    if rendered_subgoals:
        proof_block = "/- Subgoals sent to the Prover -/\n" + rendered_subgoals + "\n"

    if failures:
        detail_lines = ["Executor failed to solve some subgoals."]
        detail_lines.extend(f"subgoal {idx} failed" for idx, _ in failures)
        return ExecutionResult(
            success=False,
            stage="task_failure",
            subgoals=len(task_specs),
            detail="\n".join(detail_lines),
            attempts=attempts_total,
            proof_block=proof_block or None,
            failed_tasks=failures,
            prover_prompt_tokens=prover_prompt_tokens,
            prover_completion_tokens=prover_completion_tokens,
            prover_inference_ms=prover_inference_ms,
        )

    summary = f"Executor solved all deferred subgoals. ({len(task_specs)} subgoal(s))"
    return ExecutionResult(
        success=True,
        stage="success",
        subgoals=len(task_specs),
        detail=summary,
        attempts=attempts_total,
        proof_block=proof_block or None,
        prover_prompt_tokens=prover_prompt_tokens,
        prover_completion_tokens=prover_completion_tokens,
        prover_inference_ms=prover_inference_ms,
    )
