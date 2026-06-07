"""
01 — Single-Qubit Gates
=======================
Learn the six fundamental single-qubit gates (H, X, Y, Z, S, T),
build a one-qubit circuit for each, and visualise them as text diagrams.

Run:
    python IBM-Qiskit/01_single_qubit_gates.py
"""

from __future__ import annotations


def single_qubit_demos() -> dict:
    """Return a dict of gate-name → QuantumCircuit for every basic gate."""
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required. Install it with:  pip install qiskit"
        ) from exc

    gates = {
        "Hadamard (H)": lambda qc: qc.h(0),
        "Pauli-X (NOT)": lambda qc: qc.x(0),
        "Pauli-Y": lambda qc: qc.y(0),
        "Pauli-Z": lambda qc: qc.z(0),
        "S (√Z)": lambda qc: qc.s(0),
        "T (⁴√Z)": lambda qc: qc.t(0),
    }

    circuits = {}
    for name, apply_gate in gates.items():
        qc = QuantumCircuit(1, 1)
        apply_gate(qc)
        qc.measure(0, 0)
        circuits[name] = qc

    return circuits


def main() -> None:
    circuits = single_qubit_demos()
    for name, qc in circuits.items():
        print(f"\n{'─' * 40}")
        print(f"  {name}")
        print(f"{'─' * 40}")
        print(qc.draw(output="text"))


if __name__ == "__main__":
    main()
