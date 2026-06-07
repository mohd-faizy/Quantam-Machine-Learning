"""
03 — Bell States
================
Create all four maximally-entangled Bell states and simulate each one
to show the measurement correlation.

    |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
    |Φ⁻⟩ = (|00⟩ − |11⟩) / √2
    |Ψ⁺⟩ = (|01⟩ + |10⟩) / √2
    |Ψ⁻⟩ = (|01⟩ − |10⟩) / √2

Run:
    python IBM-Qiskit/03_bell_states.py
"""

from __future__ import annotations


def bell_circuit(label: str = "phi_plus"):
    """Build a circuit that prepares the requested Bell state.

    Args:
        label: one of 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required. Install it with:  pip install qiskit"
        ) from exc

    qc = QuantumCircuit(2, 2)

    # |Ψ⟩ states need an X on q1 before the Bell pair
    if label in {"psi_plus", "psi_minus"}:
        qc.x(1)

    qc.h(0)
    qc.cx(0, 1)

    # "minus" states get a Z phase flip
    if label in {"phi_minus", "psi_minus"}:
        qc.z(0)

    qc.measure([0, 1], [0, 1])
    return qc


def simulate(qc, shots: int = 1024) -> dict:
    """Simulate a circuit using Qiskit's built-in statevector simulator."""
    try:
        from qiskit.primitives import StatevectorSampler
    except ImportError:
        # Fallback for older Qiskit versions
        from qiskit.quantum_info import Statevector
        sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
        probs = sv.probabilities_dict()
        return {k: int(v * shots) for k, v in probs.items() if v > 0.001}

    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = result[0].data.c.get_counts()
    return dict(counts)


def main() -> None:
    labels = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
    symbols = ["Φ⁺", "Φ⁻", "Ψ⁺", "Ψ⁻"]

    for label, sym in zip(labels, symbols):
        qc = bell_circuit(label)
        counts = simulate(qc)
        print(f"\n{'─' * 40}")
        print(f"  Bell state |{sym}⟩  ({label})")
        print(f"{'─' * 40}")
        print(qc.draw(output="text"))
        print(f"  Counts: {counts}")


if __name__ == "__main__":
    main()
