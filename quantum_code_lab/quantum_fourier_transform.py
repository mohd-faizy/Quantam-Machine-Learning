"""Reusable QFT circuit implementation."""

from math import pi


def qft(n_qubits: int):
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit("Install qiskit: pip install qiskit") from exc

    circuit = QuantumCircuit(n_qubits, name="QFT")
    for target in range(n_qubits):
        circuit.h(target)
        for control in range(target + 1, n_qubits):
            circuit.cp(pi / 2 ** (control - target), control, target)
    for i in range(n_qubits // 2):
        circuit.swap(i, n_qubits - i - 1)
    return circuit


if __name__ == "__main__":
    print(qft(4).draw(output="text"))

