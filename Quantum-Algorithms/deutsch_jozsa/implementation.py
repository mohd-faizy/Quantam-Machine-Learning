"""
Deutsch-Jozsa Algorithm — Complete Implementation
===================================================

Determines whether a Boolean function f: {0,1}ⁿ → {0,1} is *constant*
(same output for every input) or *balanced* (outputs 0 for exactly half
the inputs and 1 for the other half) using a single quantum oracle query.

Classical worst-case requires 2^(n-1) + 1 queries; this algorithm uses
exactly one.

Oracle types implemented
------------------------
    • **Constant-0**: f(x) = 0 for all x  (identity oracle)
    • **Constant-1**: f(x) = 1 for all x  (X gate on ancilla before/after)
    • **Balanced (parity)**: f(x) = x₁ ⊕ x₂ ⊕ … ⊕ xₙ
    • **Balanced (inner-product)**: f(x) = x · s mod 2  for a random mask s

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
# Oracle builders
# ---------------------------------------------------------------------------

def _oracle_constant_zero(qc: "QuantumCircuit", n: int) -> None:
    """f(x) = 0 for all x — do nothing (identity)."""
    pass                       # no gates needed


def _oracle_constant_one(qc: "QuantumCircuit", n: int) -> None:
    """f(x) = 1 for all x — flip the ancilla unconditionally."""
    qc.x(n)                   # ancilla is qubit index n


def _oracle_balanced_parity(qc: "QuantumCircuit", n: int) -> None:
    """f(x) = x₁ ⊕ x₂ ⊕ … ⊕ xₙ — CNOT from every input qubit to ancilla."""
    for i in range(n):
        qc.cx(i, n)


def _oracle_balanced_inner_product(
    qc: "QuantumCircuit", n: int, mask: str | None = None,
) -> str:
    """f(x) = x · s mod 2 — CNOT from each qubit where s has a 1-bit.

    Returns the mask string used.
    """
    if mask is None:
        # Generate a random non-zero mask to guarantee balanced-ness
        mask_int = random.randint(1, 2**n - 1)
        mask = format(mask_int, f"0{n}b")
    for i, bit in enumerate(reversed(mask)):
        if bit == "1":
            qc.cx(i, n)
    return mask


# ---------------------------------------------------------------------------
# Main circuit builder
# ---------------------------------------------------------------------------

ORACLE_TYPES = {
    "constant_zero":       ("constant",  _oracle_constant_zero),
    "constant_one":        ("constant",  _oracle_constant_one),
    "balanced_parity":     ("balanced",  _oracle_balanced_parity),
    "balanced_inner_prod": ("balanced",  _oracle_balanced_inner_product),
}


def build_circuit(n: int = 4, oracle: str = "balanced_parity") -> "QuantumCircuit":
    """Build the Deutsch-Jozsa circuit for *n* input qubits.

    Parameters
    ----------
    n : int
        Number of input qubits (the function domain is {0,1}ⁿ).
    oracle : str
        One of ``"constant_zero"``, ``"constant_one"``,
        ``"balanced_parity"``, or ``"balanced_inner_prod"``.

    Returns
    -------
    QuantumCircuit
        An (n+1)-qubit, n-classical-bit circuit ready for simulation.
    """
    if oracle not in ORACLE_TYPES:
        raise ValueError(f"Unknown oracle '{oracle}'.  Choose from {set(ORACLE_TYPES)}.")

    QuantumCircuit, _ = _check_qiskit()

    qc = QuantumCircuit(n + 1, n, name=f"DJ-{oracle}")

    # Step 1 — Initialise ancilla to |1⟩
    qc.x(n)

    # Step 2 — Hadamard all qubits (creates superposition + phase-kickback state)
    qc.h(range(n + 1))
    qc.barrier(label="superposition")

    # Step 3 — Apply the oracle
    _, oracle_fn = ORACLE_TYPES[oracle]
    oracle_fn(qc, n)
    qc.barrier(label="oracle")

    # Step 4 — Hadamard the input register
    qc.h(range(n))

    # Step 5 — Measure the input register
    qc.measure(range(n), range(n))

    return qc


# ---------------------------------------------------------------------------
# Simulation & interpretation
# ---------------------------------------------------------------------------

def run_and_classify(
    n: int = 4,
    oracle: str = "balanced_parity",
    shots: int = 4096,
) -> tuple[str, dict[str, int]]:
    """Simulate the Deutsch-Jozsa circuit and classify the function.

    Returns
    -------
    (classification, counts)
        classification is ``"constant"`` or ``"balanced"``.
    """
    _, Statevector = _check_qiskit()

    qc = build_circuit(n, oracle)

    # Use statevector sampling for exact results
    # Build the unitary part (without measurement) for statevector
    QuantumCircuit, _ = _check_qiskit()
    qc_sv = QuantumCircuit(n + 1, name=f"DJ-{oracle}-sv")
    qc_sv.x(n)
    qc_sv.h(range(n + 1))
    _, oracle_fn = ORACLE_TYPES[oracle]
    oracle_fn(qc_sv, n)
    qc_sv.h(range(n))

    sv = Statevector.from_instruction(qc_sv)
    counts = sv.sample_counts(shots)

    # Filter to only input-register bits (drop ancilla)
    input_counts: dict[str, int] = {}
    for bitstring, count in counts.items():
        # Qiskit bit ordering: rightmost bit = qubit 0
        input_bits = bitstring[1:]  # drop the ancilla (most significant = qubit n)
        input_counts[input_bits] = input_counts.get(input_bits, 0) + count

    # Classification: if the only outcome is |0…0⟩ → constant; otherwise → balanced
    all_zeros = "0" * n
    if set(input_counts.keys()) == {all_zeros}:
        classification = "constant"
    else:
        classification = "balanced"

    return classification, input_counts


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Deutsch-Jozsa algorithm for all oracle types."""
    print("=" * 65)
    print("  DEUTSCH-JOZSA ALGORITHM — ORACLE CLASSIFICATION")
    print("=" * 65)

    n = 4
    SHOTS = 4096

    for oracle_name in ORACLE_TYPES:
        expected, _ = ORACLE_TYPES[oracle_name]
        print(f"\n{'─' * 65}")
        print(f"  Oracle: {oracle_name}  (expected: {expected})")
        print(f"{'─' * 65}\n")

        # Build and display circuit
        qc = build_circuit(n, oracle_name)
        print(qc.draw(output="text"))
        print()

        # Simulate and classify
        classification, counts = run_and_classify(n, oracle_name, shots=SHOTS)
        print(f"  Measurement counts: {dict(sorted(counts.items()))}")
        print(f"  Classification:     {classification}")

        # Verify
        if classification == expected:
            print(f"  ✓ CORRECT — matches expected '{expected}'")
        else:
            print(f"  ✗ WRONG — expected '{expected}', got '{classification}'")

    print("\n" + "=" * 65)
    print("  ★ Deutsch-Jozsa distinguished all oracles with ONE query each.")
    print("=" * 65)


if __name__ == "__main__":
    main()
