"""Tests for the executor with defer-based blueprints."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from minif2f_decompose.executor import (
    ProofExecutor,
    ExecutorConfig,
    ExecutionResult,
    extract_tasks,
    prepare_task_specs,
    assemble_blueprint,
    extract_proof_body,
    extract_clean_proof,
    DEFER_PRELUDE,
    TaskOutcome,
    DeferTask,
    splice_proofs,
)
from minif2f_decompose.clients import LLMClient, LeanClient, LLMResponse, CompileResult


# -----------------------------------------------------------------------------
# Test Blueprint
# -----------------------------------------------------------------------------

TEST_BLUEPRINT = """\
theorem test_let_in_type (n : ℕ) (hn : n > 0) :
    let k := n + 1
    let m := k * 2
    m > n ∧ m > k := by
  intro k m
  constructor
  · defer hn
  · defer hn
"""

# Expected subgoal proofs (what prover would return)
SUBGOAL_PROOF_1 = """\
theorem subgoal_0 (n : ℕ) (hn : n > 0) (k : ℕ) (m : ℕ) : m > n := by
  omega
"""

SUBGOAL_PROOF_2 = """\
theorem subgoal_1 (n : ℕ) (hn : n > 0) (k : ℕ) (m : ℕ) : m > k := by
  omega
"""

# The final spliced proof should look like:
EXPECTED_FINAL_PROOF = """\
theorem test_let_in_type (n : ℕ) (hn : n > 0) :
    let k := n + 1
    let m := k * 2
    m > n ∧ m > k := by
  intro k m
  constructor
  · omega
  · omega
"""

HEADER_LINES = [
    "import Mathlib",
    "import Aesop",
]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestAssembleBlueprint:
    """Test blueprint assembly."""

    def test_includes_defer_prelude(self):
        code = assemble_blueprint(TEST_BLUEPRINT, HEADER_LINES)
        assert "namespace Defer" in code
        assert "elab \"defer\"" in code

    def test_includes_imports(self):
        code = assemble_blueprint(TEST_BLUEPRINT, HEADER_LINES)
        assert "import Mathlib" in code
        assert "import Lean" in code  # Added by _dedupe_imports

    def test_includes_blueprint(self):
        code = assemble_blueprint(TEST_BLUEPRINT, HEADER_LINES)
        assert "theorem test_let_in_type" in code


class TestExtractTasks:
    """Test task extraction from compiler messages."""

    def test_extracts_defer_messages(self):
        # Simulated compiler output with [DEFER] messages
        messages = [
            {
                "severity": "info",
                "data": '[DEFER] {"goal":"m > n","context":["(n : ℕ)","(hn : n > 0)","(k : ℕ)","(m : ℕ)"],"file":"test.lean","start_byte":100,"end_byte":108,"indent":4}',
            },
            {
                "severity": "info",
                "data": '[DEFER] {"goal":"m > k","context":["(n : ℕ)","(hn : n > 0)","(k : ℕ)","(m : ℕ)"],"file":"test.lean","start_byte":120,"end_byte":128,"indent":4}',
            },
        ]
        tasks = extract_tasks(messages)
        assert len(tasks) == 2
        assert tasks[0].goal == "m > n"
        assert tasks[1].goal == "m > k"
        assert tasks[0].start_byte == 100
        assert tasks[0].end_byte == 108

    def test_ignores_non_defer_messages(self):
        messages = [
            {"severity": "info", "data": "Some other info"},
            {"severity": "error", "data": "Some error"},
        ]
        tasks = extract_tasks(messages)
        assert len(tasks) == 0


class TestPrepareTaskSpecs:
    """Test task spec preparation."""

    def test_creates_theorem_header(self):
        from minif2f_decompose.executor import DeferTask

        tasks = [
            DeferTask(
                goal="m > n",
                context_lines=["(n : ℕ)", "(hn : n > 0)", "(k : ℕ)", "(m : ℕ)"],
                start_byte=100,
                end_byte=108,
                indent=4,
            )
        ]
        specs = prepare_task_specs(tasks)
        assert len(specs) == 1
        assert "theorem subgoal_0" in specs[0].header
        assert "m > n" in specs[0].header
        assert specs[0].start_byte == 100
        assert specs[0].end_byte == 108


class TestExtractProofBody:
    """Test proof body extraction."""

    def test_extracts_body_after_by(self):
        snippet = """\
theorem foo : True := by
  trivial
"""
        body = extract_proof_body(snippet)
        assert body.strip() == "trivial"

    def test_handles_multiline(self):
        snippet = """\
theorem bar (n : ℕ) : n = n := by
  rfl
"""
        body = extract_proof_body(snippet)
        assert body.strip() == "rfl"


class TestExecutorIntegration:
    """Integration test for the full executor flow."""

    @pytest.mark.asyncio
    async def test_executor_flow(self):
        """Test the full executor flow with mocked clients."""
        # Mock LLM client
        llm = AsyncMock(spec=LLMClient)
        call_count = 0

        async def mock_llm_call(**kwargs):
            nonlocal call_count
            call_count += 1
            # Return appropriate proof for each subgoal
            if call_count == 1:
                content = f"```lean4\n{SUBGOAL_PROOF_1}\n```"
            else:
                content = f"```lean4\n{SUBGOAL_PROOF_2}\n```"
            return LLMResponse(
                content=content,
                prompt_tokens=100,
                completion_tokens=50,
                latency_ms=100.0,
            )

        llm.call = mock_llm_call

        # Mock Lean client
        lean = AsyncMock(spec=LeanClient)

        compile_call_count = 0

        async def mock_compile(**kwargs):
            nonlocal compile_call_count
            compile_call_count += 1
            code = kwargs.get("code", "")
            allow_sorry = kwargs.get("allow_sorry", True)

            # First call: blueprint compilation - return defer messages
            if compile_call_count == 1:
                return CompileResult(
                    ok=True,
                    messages=[
                        {
                            "severity": "info",
                            "data": '[DEFER] {"goal":"m > n","context":["(n : ℕ)","(hn : n > 0)","(k : ℕ)","(m : ℕ)"],"file":"test.lean","start_byte":100,"end_byte":108,"indent":4}',
                        },
                        {
                            "severity": "info",
                            "data": '[DEFER] {"goal":"m > k","context":["(n : ℕ)","(hn : n > 0)","(k : ℕ)","(m : ℕ)"],"file":"test.lean","start_byte":120,"end_byte":128,"indent":4}',
                        },
                    ],
                    error_log="",
                )
            # Subsequent calls: task verification
            return CompileResult(ok=True, messages=[], error_log="")

        lean.compile = mock_compile

        config = ExecutorConfig(
            prover_model="test-model",
            prover_sampling={},
            verify_timeout=60,
            max_prover_attempts=2,
            session_id="test-session",
        )

        executor = ProofExecutor(llm, lean, config)
        result = await executor.execute(TEST_BLUEPRINT, HEADER_LINES)

        # Check result
        assert result.success, f"Expected success but got: {result.detail}"
        assert result.stage == "done"
        assert result.subgoals == 2
        assert result.attempts == 2  # One attempt per subgoal


class TestSpliceProofs:
    """Test proof splicing back into blueprint."""

    def test_splice_single_defer(self):
        """Test splicing a single proof back."""
        # Original code with defer (ASCII only, so bytes == chars)
        original = "theorem foo : True := by\n  defer\n"
        original_bytes = original.encode("utf-8")

        # Find byte position of "defer"
        defer_start = original_bytes.index(b"defer")
        defer_end = defer_start + 5

        tasks = [
            DeferTask(
                goal="True",
                context_lines=[],
                start_byte=defer_start,
                end_byte=defer_end,
                indent=2,
            )
        ]

        outcomes = [
            TaskOutcome(idx=0, success=True, body="trivial", log="", attempts=1)
        ]

        spliced = splice_proofs(original, tasks, outcomes)
        assert "trivial" in spliced
        assert "defer" not in spliced

    def test_splice_multiple_defers(self):
        """Test splicing multiple proofs back."""
        # Blueprint with two defers (using ℕ which is 3 bytes in UTF-8)
        original = """\
theorem test (a b : ℕ) : a = a ∧ b = b := by
  constructor
  · defer
  · defer
"""
        # Find actual BYTE positions of the two "defer" occurrences
        original_bytes = original.encode("utf-8")
        first_defer_start = original_bytes.index(b"defer")
        first_defer_end = first_defer_start + 5
        second_defer_start = original_bytes.index(b"defer", first_defer_end)
        second_defer_end = second_defer_start + 5

        tasks = [
            DeferTask(
                goal="a = a",
                context_lines=["(a : ℕ)", "(b : ℕ)"],
                start_byte=first_defer_start,
                end_byte=first_defer_end,
                indent=4,
            ),
            DeferTask(
                goal="b = b",
                context_lines=["(a : ℕ)", "(b : ℕ)"],
                start_byte=second_defer_start,
                end_byte=second_defer_end,
                indent=4,
            ),
        ]

        outcomes = [
            TaskOutcome(idx=0, success=True, body="rfl", log="", attempts=1),
            TaskOutcome(idx=1, success=True, body="rfl", log="", attempts=1),
        ]

        spliced = splice_proofs(original, tasks, outcomes)
        assert "defer" not in spliced
        assert spliced.count("rfl") == 2


class TestExtractCleanProof:
    """Test clean proof extraction (without DEFER_PRELUDE)."""

    def test_removes_defer_namespace(self):
        """Test that Defer namespace is removed from final proof."""
        # Assemble a blueprint (includes DEFER_PRELUDE)
        code_with_prelude = assemble_blueprint(TEST_BLUEPRINT, HEADER_LINES)

        # Verify the code has the Defer namespace
        assert "namespace Defer" in code_with_prelude
        assert "end Defer" in code_with_prelude

        # Extract clean proof
        clean = extract_clean_proof(code_with_prelude, HEADER_LINES)

        # Verify Defer namespace is removed
        assert "namespace Defer" not in clean
        assert "end Defer" not in clean
        assert "elab \"defer\"" not in clean

    def test_preserves_theorem(self):
        """Test that the theorem is preserved in clean proof."""
        code_with_prelude = assemble_blueprint(TEST_BLUEPRINT, HEADER_LINES)
        clean = extract_clean_proof(code_with_prelude, HEADER_LINES)

        # Theorem should be preserved
        assert "theorem test_let_in_type" in clean
        assert "m > n ∧ m > k" in clean

    def test_preserves_imports(self):
        """Test that imports are preserved in clean proof."""
        code_with_prelude = assemble_blueprint(TEST_BLUEPRINT, HEADER_LINES)
        clean = extract_clean_proof(code_with_prelude, HEADER_LINES)

        # Standard imports should be there
        assert "import Mathlib" in clean
        assert "import Lean" in clean


class TestRealVerification:
    """Integration tests with real Lean verification server."""

    @pytest.mark.asyncio
    async def test_blueprint_compiles_with_defer(self):
        """Test that our blueprint actually compiles with defer tactic."""
        import os
        from dotenv import load_dotenv

        load_dotenv("/Volumes/Disk2/Lean/lean-agent/.env")

        url = os.getenv("VERIFICATION_URL", "")
        key = os.getenv("VERIFICATION_KEY", "")
        if not url:
            pytest.skip("VERIFICATION_URL not set")

        from minif2f_decompose.clients import LeanClient

        lean = LeanClient(url, key)

        # Assemble the blueprint with defer prelude
        code = assemble_blueprint(TEST_BLUEPRINT, HEADER_LINES)

        # Compile with sorries allowed (defer creates sorries)
        result = await lean.compile(
            code=code,
            timeout=120,
            allow_sorry=True,
            snippet_id="test-blueprint",
        )

        assert result.ok, f"Blueprint failed to compile: {result.error_log}"

        # Check we got defer messages
        defer_messages = [
            m for m in result.messages
            if m.get("severity") == "info" and "[DEFER]" in str(m.get("data", ""))
        ]
        assert len(defer_messages) >= 1, "Expected at least one [DEFER] message"

    @pytest.mark.asyncio
    async def test_spliced_proof_compiles(self):
        """Test that splicing proofs back produces valid Lean code."""
        import os
        from dotenv import load_dotenv

        load_dotenv("/Volumes/Disk2/Lean/lean-agent/.env")

        url = os.getenv("VERIFICATION_URL", "")
        key = os.getenv("VERIFICATION_KEY", "")
        if not url:
            pytest.skip("VERIFICATION_URL not set")

        from minif2f_decompose.clients import LeanClient

        lean = LeanClient(url, key)

        # Step 1: Compile blueprint to get defer messages
        code_with_prelude = assemble_blueprint(TEST_BLUEPRINT, HEADER_LINES)

        compile_result = await lean.compile(
            code=code_with_prelude,
            timeout=120,
            allow_sorry=True,
            snippet_id="test-step1",
        )
        assert compile_result.ok, f"Blueprint failed: {compile_result.error_log}"

        # Step 2: Extract tasks
        tasks = extract_tasks(compile_result.messages, code_with_prelude)
        assert len(tasks) == 2, f"Expected 2 tasks, got {len(tasks)}"

        # Step 3: Create fake outcomes with "omega" as proof
        outcomes = [
            TaskOutcome(idx=0, success=True, body="omega", log="", attempts=1),
            TaskOutcome(idx=1, success=True, body="omega", log="", attempts=1),
        ]

        # Step 4: Splice proofs back
        from minif2f_decompose.executor import splice_proofs
        spliced = splice_proofs(code_with_prelude, tasks, outcomes)

        # Verify defer is replaced
        assert "omega" in spliced, "Proof body not spliced in"

        # Step 5: Final verification - should compile without sorries
        final_result = await lean.compile(
            code=spliced,
            timeout=120,
            allow_sorry=False,
            snippet_id="test-final",
        )

        assert final_result.ok, f"Spliced proof failed to compile: {final_result.error_log}\n\nSpliced code:\n{spliced}"

    @pytest.mark.asyncio
    async def test_executor_returns_clean_proof(self):
        """Test that the executor returns a clean proof without DEFER_PRELUDE."""
        import os
        from dotenv import load_dotenv
        from openai import AsyncOpenAI

        load_dotenv("/Volumes/Disk2/Lean/lean-agent/.env")

        url = os.getenv("VERIFICATION_URL", "")
        key = os.getenv("VERIFICATION_KEY", "")
        prover_url = os.getenv("PROVER_BASE_URL", "")
        prover_key = os.getenv("PROVER_KEY", "")
        prover_model = os.getenv("PROVER_MODEL", "")
        if not url or not prover_url:
            pytest.skip("VERIFICATION_URL or PROVER_BASE_URL not set")

        from minif2f_decompose.clients import LeanClient, LLMClient, create_httpx_client

        lean = LeanClient(url, key)
        openai_client = AsyncOpenAI(
            api_key=prover_key,
            base_url=prover_url,
            http_client=create_httpx_client(),
        )
        llm = LLMClient(openai_client, max_parallel=8)

        config = ExecutorConfig(
            prover_model=prover_model,
            prover_sampling={"temperature": 0.0, "max_tokens": 2048},
            verify_timeout=120,
            max_prover_attempts=3,
            session_id="test-clean-proof",
        )

        executor = ProofExecutor(llm, lean, config)
        result = await executor.execute(TEST_BLUEPRINT, HEADER_LINES)

        # If successful, verify the proof_block is clean
        if result.success:
            assert result.proof_block is not None
            assert "namespace Defer" not in result.proof_block
            assert "end Defer" not in result.proof_block
            assert "elab \"defer\"" not in result.proof_block
            # Should have the theorem
            assert "theorem test_let_in_type" in result.proof_block
            # Should have the proof bodies (omega)
            assert "omega" in result.proof_block
            # Should NOT have defer
            assert "defer" not in result.proof_block


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
