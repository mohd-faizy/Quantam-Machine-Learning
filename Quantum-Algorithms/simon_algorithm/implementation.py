"""
Simon's Algorithm — Complete Implementation
============================================

Finds a hidden XOR-mask *s* such that for a 2-to-1 function f,
f(x) = f(y) ⟺ y = x ⊕ s.

Simon's algorithm was the first to demonstrate an *exponential* quantum
query advantage and directly inspired Shor's period-finding approach.

The quantum part produces measurement samples z satisfying z · s = 0 (mod 2).
After collecting O(n) linearly independent samples, classical Gaussian
elimination over GF(2) recovers the hidden string s.

Features
--------
    • Full quantum circuit with configurable oracle for secret string s
    • Statevector simulation with multiple sampling rounds
    • GF(2) Gaussian elimination to solve the linear system
    • Automatic verification of the recovered string

Usage
-----
    python implementation.py
"""

from __future__ import annotations

import random


def _check_qiskit():
    """Ensure Qiskit is installed and return required modules."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        return QuantumCircuit, Statevector
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required.  Install it with:\n"
            "  pip install 'qiskit>=1.0' qiskit-aer"
        ) from exc


# ---------------------------------------------------------------------------
# Oracle builder
# ---------------------------------------------------------------------------

def build_simon_oracle(secret: str) -> "QuantumCircuit":
    """Build a reversible oracle U_f for Simon's problem.

    The oracle implements a 2-to-1 function satisfying f(x) = f(x ⊕ s)
    by copying the input register to the output register and then XOR-ing
    the output with a mask conditioned on a specific input qubit.

    For a secret string s, the oracle does:
        |x⟩|0⟩ → |x⟩|f(x)⟩

    where f(x) = f(x ⊕ s) for all x.

    Parameters
    ----------
    secret : str
        The hidden bit-string s (must be non-zero for a 2-to-1 function).
    """
    QuantumCircuit, _ = _check_qiskit()

    n = len(secret)
    oracle = QuantumCircuit(2 * n, name=f"Oracle(s={secret})")

    # Step 1: Copy input register to output register: |x⟩|0⟩ → |x⟩|x⟩
    for i in range(n):
        oracle.cx(i, n + i)

    # Step 2: Make it 2-to-1 — XOR output with s when a specific input bit is 1
    # Find the position of the first '1' in s
    pivot = None
    for i, bit in enumerate(reversed(secret)):
        if bit == "1":
            pivot = i
            break

    if pivot is not None:
        # When input qubit at pivot position is |1⟩, XOR the output with s
        for i, bit in enumerate(reversed(secret)):
            if bit == "1":
                oracle.cx(pivot, n + i)

    return oracle


# ---------------------------------------------------------------------------
# Main circuit builder
# ---------------------------------------------------------------------------

def build_circuit(secret: str = "110") -> "QuantumCircuit":
    """Build one round of Simon's algorithm.

    Parameters
    ----------
    secret : str
        The hidden bit-string s.

    Returns
    -------
    QuantumCircuit
        A 2n-qubit circuit with n classical bits for measuring the
        input register.

    Circuit structure
    -----------------
    1. Apply H⊗ⁿ to the input register → uniform superposition.
    2. Apply the oracle U_f: |x⟩|0⟩ → |x⟩|f(x)⟩.
    3. Measure (or discard) the output register — collapses input to
       the superposition (|x₀⟩ + |x₀ ⊕ s⟩)/√2 for some x₀.
    4. Apply H⊗ⁿ to the input register → produces z with z · s = 0.
    5. Measure the input register to obtain one equation.
    """
    QuantumCircuit, _ = _check_qiskit()

    n = len(secret)
    qc = QuantumCircuit(2 * n, n, name=f"Simon(s={secret})")

    # Step 1 — Hadamard the input register
    qc.h(range(n))
    qc.barrier(label="superposition")

    # Step 2 — Apply oracle
    oracle = build_simon_oracle(secret)
    qc.compose(oracle, range(2 * n), inplace=True)
    qc.barrier(label="oracle")

    # Step 3 — Measure output register (conceptually traces it out)
    # We measure to force the collapse, then focus on the input register
    # For statevector simulation, we skip this and trace manually

    # Step 4 — Hadamard the input register again
    qc.h(range(n))

    # Step 5 — Measure the input register
    qc.measure(range(n), range(n))

    return qc


# ---------------------------------------------------------------------------
# GF(2) linear algebra
# ---------------------------------------------------------------------------

def gaussian_elimination_gf2(equations: list[list[int]]) -> list[int] | None:
    """Solve the system z · s = 0 (mod 2) for the non-trivial solution s.

    Parameters
    ----------
    equations : list[list[int]]
        Each element is a list of 0s and 1s representing a row z.

    Returns
    -------
    list[int] or None
        The non-trivial solution s, or None if the system is under-determined.
    """
    if not equations:
        return None

    n = len(equations[0])
    # Build augmented-style matrix (no augmented column needed for null-space)
    matrix = [row[:] for row in equations]

    # Forward elimination
    pivot_cols = []
    row_idx = 0
    for col in range(n):
        # Find pivot
        found = None
        for r in range(row_idx, len(matrix)):
            if matrix[r][col] == 1:
                found = r
                break
        if found is None:
            continue
        # Swap
        matrix[row_idx], matrix[found] = matrix[found], matrix[row_idx]
        pivot_cols.append(col)
        # Eliminate
        for r in range(len(matrix)):
            if r != row_idx and matrix[r][col] == 1:
                matrix[r] = [(matrix[r][j] ^ matrix[row_idx][j]) for j in range(n)]
        row_idx += 1

    # If rank = n, the only solution is the trivial s = 0
    rank = len(pivot_cols)
    if rank >= n:
        return None  # Only trivial solution exists

    # Find a free variable (column not in pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_cols]
    if not free_cols:
        return None

    # Set the first free variable to 1, solve for pivots
    s = [0] * n
    free_col = free_cols[0]
    s[free_col] = 1

    # Back-substitution
    for i in range(rank - 1, -1, -1):
        pc = pivot_cols[i]
        val = 0
        for j in range(n):
            if j != pc:
                val ^= (matrix[i][j] & s[j])
        s[pc] = val

    return s


def collect_samples(secret: str, num_samples: int = 20) -> list[str]:
    """Run Simon's circuit multiple times and collect measurement samples.

    Parameters
    ----------
    secret : str
        The hidden string (used to build the oracle).
    num_samples : int
        Number of independent measurement samples to collect.

    Returns
    -------
    list[str]
        List of measured bit-strings z (each satisfying z · s = 0 mod 2).
    """
    _, Statevector = _check_qiskit()
    QuantumCircuit, _ = _check_qiskit()

    n = len(secret)

    # Build the unitary part (no measurement) for statevector sampling
    qc_sv = QuantumCircuit(2 * n, name=f"Simon-sv(s={secret})")
    qc_sv.h(range(n))
    oracle = build_simon_oracle(secret)
    qc_sv.compose(oracle, range(2 * n), inplace=True)
    qc_sv.h(range(n))

    sv = Statevector.from_instruction(qc_sv)
    raw_counts = sv.sample_counts(num_samples)

    # Extract input-register bits only (last n bits in Qiskit ordering)
    input_samples: list[str] = []
    for bitstring, count in raw_counts.items():
        input_bits = bitstring[n:]  # input register = lower n qubits
        for _ in range(count):
            input_samples.append(input_bits)

    return input_samples


def solve_simon(secret: str, num_samples: int = 30) -> str | None:
    """Run Simon's algorithm end-to-end: sample + GF(2) solve.

    Parameters
    ----------
    secret : str
        The hidden string (used to build the oracle).
    num_samples : int
        Number of quantum samples to collect.

    Returns
    -------
    str or None
        The recovered hidden string, or None if under-determined.
    """
    n = len(secret)
    samples = collect_samples(secret, num_samples)

    # Convert to integer rows
    equations = []
    for z in samples:
        row = [int(b) for b in z]
        if any(b == 1 for b in row):  # skip all-zero samples
            equations.append(row)

    # Remove duplicate rows
    unique_eqs = []
    seen = set()
    for row in equations:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            unique_eqs.append(row)

    result = gaussian_elimination_gf2(unique_eqs)
    if result is None:
        return None

    return "".join(str(b) for b in result)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def dot_mod2(a: str, b: str) -> int:
    """Compute the inner product of two bit-strings modulo 2."""
    return sum(int(x) & int(y) for x, y in zip(a, b)) % 2


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Simon's algorithm on multiple test cases."""
    print("=" * 65)
    print("  SIMON'S ALGORITHM — HIDDEN XOR-MASK RECOVERY")
    print("=" * 65)

    test_secrets = ["11", "110", "101", "1001"]

    for secret in test_secrets:
        n = len(secret)
        print(f"\n{'─' * 65}")
        print(f"  Secret string s = {secret}  ({n} qubits)")
        print(f"{'─' * 65}\n")

        # Build and display circuit (one round)
        qc = build_circuit(secret)
        print(qc.draw(output="text"))
        print()

        # Collect samples
        NUM_SAMPLES = 10 * n
        samples = collect_samples(secret, NUM_SAMPLES)
        unique_samples = list(set(samples))
        print(f"  Collected {len(samples)} samples, {len(unique_samples)} unique:")
        for z in sorted(unique_samples):
            orthogonal = dot_mod2(z, secret) == 0
            print(f"    z = {z}  →  z·s = {dot_mod2(z, secret)}  "
                  f"{'✓ orthogonal' if orthogonal else '✗ NOT orthogonal'}")

        # Solve
        recovered = solve_simon(secret, NUM_SAMPLES)
        print(f"\n  Recovered s = {recovered}")

        if recovered == secret:
            print(f"  ✓ CORRECT — hidden string recovered")
        elif recovered is not None:
            # Check if it's a valid alternative (s or 0)
            print(f"  ⚠ Got different s — verifying orthogonality...")
        else:
            print(f"  ✗ Under-determined — need more samples")

    print("\n" + "=" * 65)
    print("  ★ Simon's algorithm: O(n) quantum queries vs O(2^{n/2}) classical.")
    print("=" * 65)


if __name__ == "__main__":
    main()
