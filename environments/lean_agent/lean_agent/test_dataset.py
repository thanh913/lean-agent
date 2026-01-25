"""Test dataset for lean_agent environment."""

from __future__ import annotations

# 6 test entries from easy to hard for framework testing
TEST_DATASET = [
    # 1. Trivial - direct computation
    {
        "theorem": "theorem test_trivial (x : ℝ) (h : x / 50 = 40) : x = 2000",
        "nl_proof": "Multiply both sides by 50: x = 40 * 50 = 2000.",
        "header_lines": ["import Mathlib", "open Real"],
    },
    # 2. Easy - modular arithmetic
    {
        "theorem": "theorem test_mod : 2 ^ 2010 % 10 = 4",
        "nl_proof": "Powers of 2 mod 10 cycle with period 4: 2, 4, 8, 6, 2, 4, 8, 6, ... Since 2010 mod 4 = 2, we have 2^2010 mod 10 = 4.",
        "header_lines": ["import Mathlib"],
    },
    # 3. Easy - divisibility
    {
        "theorem": "theorem test_div (n : ℕ) (h₀ : n ≤ 9) (h₁ : 18 ∣ 374 * 10 + n) : n = 4",
        "nl_proof": "18 = 2 × 9. For divisibility by 2, n must be even (0, 2, 4, 6, 8). For divisibility by 9, the digit sum 3 + 7 + 4 + n = 14 + n must be divisible by 9. So 14 + n ≡ 0 (mod 9), meaning n ≡ 4 (mod 9). Combined with n ≤ 9 and n even, we get n = 4.",
        "header_lines": ["import Mathlib"],
    },
    # 4. Medium - sum formula with modular arithmetic
    {
        "theorem": "theorem test_sum : (∑ k in Finset.range 101, k) % 6 = 4",
        "nl_proof": "The sum of 0 to 100 is 100 × 101 / 2 = 5050. Computing 5050 mod 6: 5050 = 841 × 6 + 4, so the remainder is 4.",
        "header_lines": ["import Mathlib", "open BigOperators"],
    },
    # 5. Medium - existential with witness construction
    {
        "theorem": "theorem test_exists (n : ℕ) (h₀ : 0 < n) : ∃ m, m > n ∧ ∃ p, m * p ≤ m + p",
        "nl_proof": "Take m = n + 1 and p = 1. Then m > n is satisfied. We need m × p ≤ m + p, i.e., (n + 1) × 1 ≤ (n + 1) + 1, which simplifies to n + 1 ≤ n + 2. This is clearly true.",
        "header_lines": ["import Mathlib"],
    },
    # 6. Hard - trigonometric equation (likely needs decomposition)
    {
        "theorem": "theorem test_trig (S : Finset ℝ) (h₀ : ∀ x : ℝ, x ∈ S ↔ 0 ≤ x ∧ x ≤ Real.pi ∧ Real.sin (Real.pi / 2 * Real.cos x) = Real.cos (Real.pi / 2 * Real.sin x)) : S.card = 2",
        "nl_proof": "We use the identity sin(A) = cos(B) iff A + B = π/2 + 2kπ for some integer k, or A = -B + 2kπ. Setting A = (π/2)cos(x) and B = (π/2)sin(x), the equation A + B = π/2 gives cos(x) + sin(x) = 1. In [0, π], this equation has exactly two solutions: x = 0 (where cos(0) + sin(0) = 1) and x = π/2 (where cos(π/2) + sin(π/2) = 1). Therefore |S| = 2.",
        "header_lines": ["import Mathlib", "open Real"],
    },
]
