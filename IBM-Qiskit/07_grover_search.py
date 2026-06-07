"""
07 — Grover's Search Algorithm
===============================
Search for the marked state |11⟩ in a 2-qubit search space.
One Grover iteration is enough for 2 qubits (the success probability
is exactly 100 %).

Circuit breakdown:
    1. Superposition — H on both qubits.
    2. Oracle — CZ marks |11⟩ with a phase flip.
    3. Diffusion — Reflects about the mean amplitude.

Run:
    python IBM-Qiskit/07_grover_search.py
"""

from __future__ import annotations


def grover_circuit(marked: str = "11"):
    """Build a 2-qubit Grover circuit that searches for |marked⟩.

    Only the |11⟩ oracle is implemented for this educational demo.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required. Install it with:  pip install qiskit"
        ) from exc

    if marked != "11":
        raise ValueError("This compact demo implements the |11⟩ oracle only.")

    qc = QuantumCircuit(2, 2)

    # --- Step 1: Superposition ---
    qc.h([0, 1])
    qc.barrier()

    # --- Step 2: Oracle — phase-flip |11⟩ ---
    qc.cz(0, 1)
    qc.barrier()

    # --- Step 3: Diffusion operator ---
    qc.h([0, 1])
    qc.z([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    qc.barrier()

    # --- Measurement ---
    qc.measure([0, 1], [0, 1])

    return qc


def simulate(qc, shots: int = 1024) -> dict:
    """Simulate using Statevector."""
    from qiskit.quantum_info import Statevector

    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_no_meas)
    probs = sv.probabilities_dict()
    return {k: int(v * shots) for k, v in probs.items() if v > 0.001}


def main() -> None:
    qc = grover_circuit()
    counts = simulate(qc)

    print("Grover's Search — 2-qubit |11⟩ oracle")
    print("=" * 45)
    print(qc.draw(output="text"))
    print()
    print(f"Measurement results: {counts}")
    print(f"Marked state |11⟩ found with probability: "
          f"{counts.get('11', 0) / sum(counts.values()) * 100:.1f}%")


if __name__ == "__main__":
    main()
