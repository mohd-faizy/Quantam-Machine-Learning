"""
06 — Deutsch-Jozsa Algorithm
=============================
Determine whether a black-box function is **constant** (same output for
all inputs) or **balanced** (returns 0 for half and 1 for the other half)
using only a single query.

This demo uses a balanced parity oracle: f(x) = x₀ ⊕ x₁ ⊕ ... ⊕ xₙ₋₁.
The algorithm always measures a non-zero string ⟹ balanced.

Run:
    python IBM-Qiskit/06_deutsch_jozsa.py
"""

from __future__ import annotations


def deutsch_jozsa_circuit(n: int = 3, oracle_type: str = "balanced"):
    """Build the Deutsch-Jozsa circuit.

    Args:
        n: number of input qubits.
        oracle_type: 'balanced' (parity) or 'constant' (identity / X-flip).
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required. Install it with:  pip install qiskit"
        ) from exc

    qc = QuantumCircuit(n + 1, n)

    # --- Initialisation ---
    qc.x(n)               # ancilla to |1⟩
    qc.h(range(n + 1))    # Hadamard on all qubits
    qc.barrier()

    # --- Oracle ---
    if oracle_type == "balanced":
        # Parity oracle: every input qubit is CNOTed to the ancilla
        for qubit in range(n):
            qc.cx(qubit, n)
    elif oracle_type == "constant":
        # Constant oracle f(x) = 0: do nothing (identity)
        pass
    else:
        raise ValueError(f"Unknown oracle type: {oracle_type!r}")
    qc.barrier()

    # --- Decode ---
    qc.h(range(n))
    qc.measure(range(n), range(n))

    return qc


def simulate(qc, n_query: int = 3, shots: int = 1024) -> dict:
    """Simulate and return counts for the query register only."""
    from qiskit.quantum_info import Statevector, partial_trace

    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_no_meas)
    rho = partial_trace(sv, [n_query])
    probs = rho.probabilities_dict()
    return {k: int(v * shots) for k, v in probs.items() if v > 0.001}


def main() -> None:
    for oracle in ["balanced", "constant"]:
        qc = deutsch_jozsa_circuit(n=3, oracle_type=oracle)
        counts = simulate(qc)
        result_bits = max(counts, key=counts.get)
        verdict = "CONSTANT" if all(b == "0" for b in result_bits) else "BALANCED"

        print(f"\n{'─' * 45}")
        print(f"  Oracle: {oracle}  →  Algorithm says: {verdict}")
        print(f"{'─' * 45}")
        print(qc.draw(output="text"))
        print(f"  Measurement: {counts}")


if __name__ == "__main__":
    main()
