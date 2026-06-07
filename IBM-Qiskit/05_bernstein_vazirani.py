"""
05 — Bernstein-Vazirani Algorithm
==================================
Recover a hidden bit-string **s** from a black-box oracle that computes
f(x) = s · x (mod 2) in a single query — exponentially faster than
classical approaches.

Example: secret = "1011" → algorithm finds "1011" in one shot.

Run:
    python IBM-Qiskit/05_bernstein_vazirani.py
"""

from __future__ import annotations


def bernstein_vazirani_circuit(secret: str = "1011"):
    """Build the Bernstein-Vazirani circuit for a given secret string.

    Args:
        secret: binary string representing the hidden bit-string s.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required. Install it with:  pip install qiskit"
        ) from exc

    n = len(secret)
    qc = QuantumCircuit(n + 1, n)

    # --- Initialisation ---
    # Put the ancilla qubit in |−⟩
    qc.x(n)
    qc.h(range(n + 1))
    qc.barrier()

    # --- Oracle: f(x) = s · x mod 2 ---
    # For each '1' bit in the secret, apply CNOT from that qubit to ancilla
    for i, bit in enumerate(reversed(secret)):
        if bit == "1":
            qc.cx(i, n)
    qc.barrier()

    # --- Hadamard on query register to decode ---
    qc.h(range(n))
    qc.measure(range(n), range(n))

    return qc


def simulate(qc, shots: int = 1024) -> dict:
    """Simulate using Statevector-based approach."""
    try:
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise SystemExit("Qiskit is required.") from exc

    # Remove measurements, get statevector, then sample
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_no_meas)
    probs = sv.probabilities_dict()
    return {k: int(v * shots) for k, v in probs.items() if v > 0.001}


def main() -> None:
    secret = "1011"
    qc = bernstein_vazirani_circuit(secret)
    counts = simulate(qc)

    print("Bernstein-Vazirani Algorithm")
    print("=" * 45)
    print(f"Hidden string: {secret}")
    print()
    print(qc.draw(output="text"))
    print()
    print(f"Measurement results: {counts}")
    print(f"Recovered string:    {max(counts, key=counts.get)}")


if __name__ == "__main__":
    main()
