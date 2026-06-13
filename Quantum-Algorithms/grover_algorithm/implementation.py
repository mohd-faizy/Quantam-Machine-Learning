"""
Grover's Search Algorithm — Complete Implementation
====================================================

Searches an unstructured database of N = 2ⁿ items for a marked element
using only O(√N) oracle queries, achieving a provably optimal quadratic
speedup over classical brute-force search.

Features
--------
    • General n-qubit implementation (not hardcoded to any specific state)
    • Configurable marked state(s) — works with single or multiple targets
    • Proper oracle construction using multi-controlled-Z decomposition
    • Correct iteration count: ⌊π/4 · √(N/M)⌋ for M marked states
    • Simulation with success probability analysis
    • Amplitude evolution tracking across iterations

Usage
-----
    python implementation.py
"""

from __future__ import annotations

import math


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
# Oracle & Diffusion operators
# ---------------------------------------------------------------------------

def build_oracle(n: int, marked_states: list[str]) -> "QuantumCircuit":
    """Build the Grover oracle that flips the phase of marked states.

    The oracle applies the transformation:
        |x⟩ → (−1)^{f(x)} |x⟩

    where f(x) = 1 if x is a marked state, 0 otherwise.

    Parameters
    ----------
    n : int
        Number of qubits.
    marked_states : list[str]
        List of bit-strings to mark (e.g., ["101", "011"]).
    """
    QuantumCircuit, _ = _check_qiskit()

    oracle = QuantumCircuit(n, name="Oracle")

    for target in marked_states:
        if len(target) != n:
            raise ValueError(f"Marked state '{target}' has wrong length (expected {n}).")

        # Flip qubits where the target has '0' (so CZ fires on the target pattern)
        flip_qubits = []
        for i, bit in enumerate(reversed(target)):
            if bit == "0":
                oracle.x(i)
                flip_qubits.append(i)

        # Multi-controlled-Z: flip phase when all qubits are |1⟩
        if n == 1:
            oracle.z(0)
        elif n == 2:
            oracle.cz(0, 1)
        else:
            # Decompose MCZ as: H on target → MCX → H on target
            oracle.h(n - 1)
            oracle.mcx(list(range(n - 1)), n - 1)
            oracle.h(n - 1)

        # Undo the flips
        for i in flip_qubits:
            oracle.x(i)

    return oracle


def build_diffusion(n: int) -> "QuantumCircuit":
    """Build the Grover diffusion operator (reflection about the mean).

    Implements the transformation:
        D = 2|s⟩⟨s| − I

    where |s⟩ = H⊗ⁿ|0⟩ is the uniform superposition.

    This is equivalent to: H⊗ⁿ · (2|0⟩⟨0| − I) · H⊗ⁿ
    """
    QuantumCircuit, _ = _check_qiskit()

    diffusion = QuantumCircuit(n, name="Diffusion")

    # H⊗ⁿ
    diffusion.h(range(n))

    # 2|0⟩⟨0| − I  =  conditional phase flip on |0…0⟩
    # Flip all qubits, apply MCZ, flip back
    diffusion.x(range(n))

    if n == 1:
        diffusion.z(0)
    elif n == 2:
        diffusion.cz(0, 1)
    else:
        diffusion.h(n - 1)
        diffusion.mcx(list(range(n - 1)), n - 1)
        diffusion.h(n - 1)

    diffusion.x(range(n))

    # H⊗ⁿ
    diffusion.h(range(n))

    return diffusion


# ---------------------------------------------------------------------------
# Main circuit builder
# ---------------------------------------------------------------------------

def optimal_iterations(n: int, num_marked: int = 1) -> int:
    """Compute the optimal number of Grover iterations.

    Returns ⌊π/4 · √(N/M)⌋ where N = 2ⁿ and M = num_marked.
    """
    N = 2 ** n
    return max(1, math.floor(math.pi / 4 * math.sqrt(N / num_marked)))


def build_circuit(
    n: int = 3,
    marked_states: list[str] | None = None,
    num_iterations: int | None = None,
) -> "QuantumCircuit":
    """Build the complete Grover search circuit.

    Parameters
    ----------
    n : int
        Number of qubits (search space size = 2ⁿ).
    marked_states : list[str], optional
        States to search for.  Defaults to ["1" * n] (all-ones state).
    num_iterations : int, optional
        Number of Grover iterations.  Defaults to the optimal count.

    Returns
    -------
    QuantumCircuit
        An n-qubit circuit with measurements.
    """
    QuantumCircuit, _ = _check_qiskit()

    if marked_states is None:
        marked_states = ["1" * n]

    if num_iterations is None:
        num_iterations = optimal_iterations(n, len(marked_states))

    qc = QuantumCircuit(n, n, name=f"Grover(n={n})")

    # Step 1 — Uniform superposition
    qc.h(range(n))

    # Step 2 — Repeat Oracle + Diffusion
    oracle = build_oracle(n, marked_states)
    diffusion = build_diffusion(n)

    for k in range(num_iterations):
        qc.barrier(label=f"iter {k+1}")
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)

    # Step 3 — Measure
    qc.barrier(label="measure")
    qc.measure(range(n), range(n))

    return qc


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_grover(
    n: int = 3,
    marked_states: list[str] | None = None,
    shots: int = 8192,
) -> tuple[str, dict[str, int], float]:
    """Simulate Grover's algorithm and return results.

    Returns
    -------
    (top_result, counts, success_probability)
    """
    _, Statevector = _check_qiskit()
    QuantumCircuit, _ = _check_qiskit()

    if marked_states is None:
        marked_states = ["1" * n]

    num_iters = optimal_iterations(n, len(marked_states))

    # Build the unitary part (no measurement)
    qc_sv = QuantumCircuit(n, name=f"Grover-sv")
    qc_sv.h(range(n))

    oracle = build_oracle(n, marked_states)
    diffusion = build_diffusion(n)

    for _ in range(num_iters):
        qc_sv.compose(oracle, inplace=True)
        qc_sv.compose(diffusion, inplace=True)

    sv = Statevector.from_instruction(qc_sv)
    counts = sv.sample_counts(shots)

    # Success probability
    success_count = sum(counts.get(m, 0) for m in marked_states)
    success_prob = success_count / shots

    top_result = max(counts, key=counts.get)

    return top_result, counts, success_prob


def track_amplitude_evolution(
    n: int = 3,
    marked_state: str = "111",
    max_iters: int | None = None,
) -> list[float]:
    """Track the probability of the marked state across iterations.

    Returns a list of probabilities, one per iteration (starting from 0).
    """
    _, Statevector = _check_qiskit()
    QuantumCircuit, _ = _check_qiskit()

    if max_iters is None:
        max_iters = optimal_iterations(n, 1) + 3  # a few extra to show overshooting

    marked_index = int(marked_state, 2)

    oracle = build_oracle(n, [marked_state])
    diffusion = build_diffusion(n)

    probabilities = []

    qc = QuantumCircuit(n)
    qc.h(range(n))

    sv = Statevector.from_instruction(qc)
    probabilities.append(abs(sv[marked_index]) ** 2)

    for k in range(max_iters):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
        sv = Statevector.from_instruction(qc)
        probabilities.append(abs(sv[marked_index]) ** 2)

    return probabilities


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Grover's search on multiple configurations."""
    print("=" * 70)
    print("  GROVER'S SEARCH ALGORITHM — AMPLITUDE AMPLIFICATION")
    print("=" * 70)

    SHOTS = 8192

    # --- Test case 1: Single marked state on 3 qubits ---
    test_cases = [
        (2, ["11"],  "2 qubits, 1 marked state"),
        (3, ["101"], "3 qubits, 1 marked state"),
        (3, ["010", "110"], "3 qubits, 2 marked states"),
        (4, ["1010"], "4 qubits, 1 marked state"),
    ]

    for n, marked, description in test_cases:
        N = 2 ** n
        num_iters = optimal_iterations(n, len(marked))

        print(f"\n{'─' * 70}")
        print(f"  {description}")
        print(f"  N={N}, marked={marked}, optimal iterations={num_iters}")
        print(f"{'─' * 70}\n")

        # Build and display circuit
        qc = build_circuit(n, marked)
        print(qc.draw(output="text"))
        print()

        # Simulate
        top_result, counts, success_prob = simulate_grover(n, marked, SHOTS)

        # Display top results
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        print(f"  Top measurement results ({SHOTS:,} shots):")
        for bitstring, count in sorted_counts[:8]:
            marker = " ◄ MARKED" if bitstring in marked else ""
            bar = "█" * int(40 * count / SHOTS)
            print(f"    |{bitstring}⟩  {count:5d}  ({count/SHOTS:6.2%})  {bar}{marker}")

        print(f"\n  Success probability: {success_prob:.4%}")

        if top_result in marked:
            print(f"  ✓ Top result |{top_result}⟩ is a marked state")
        else:
            print(f"  ✗ Top result |{top_result}⟩ is NOT a marked state")

    # --- Amplitude evolution for 3-qubit case ---
    print(f"\n{'─' * 70}")
    print(f"  AMPLITUDE EVOLUTION — 3 qubits, target |101⟩")
    print(f"{'─' * 70}\n")

    probs = track_amplitude_evolution(3, "101", max_iters=6)
    opt_iter = optimal_iterations(3, 1)

    for k, p in enumerate(probs):
        bar = "█" * int(50 * p)
        marker = " ◄ optimal" if k == opt_iter else ""
        print(f"  iter {k}: P(|101⟩) = {p:.6f}  {bar}{marker}")

    print("\n" + "=" * 70)
    print("  ★ Grover achieves O(√N) search — quadratic quantum speedup.")
    print("=" * 70)


if __name__ == "__main__":
    main()
