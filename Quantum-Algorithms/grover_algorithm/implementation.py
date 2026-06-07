"""Grover search demo for a two-qubit search space."""

from __future__ import annotations


def build_circuit(marked_state: str = "11"):
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit("Install qiskit to run this demo: pip install qiskit") from exc

    if marked_state != "11":
        raise ValueError("This compact demo implements the |11> phase oracle.")

    qc = QuantumCircuit(2, 2)
    qc.h([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    qc.z([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    qc.measure([0, 1], [0, 1])
    return qc


def main() -> None:
    qc = build_circuit()
    print(qc.draw(output="text"))


if __name__ == "__main__":
    main()

