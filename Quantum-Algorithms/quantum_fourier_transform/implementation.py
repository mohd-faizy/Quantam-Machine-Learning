"""Quantum Fourier Transform circuit builder."""

from math import pi


def build_qft(n: int = 3):
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit("Install qiskit to run this demo: pip install qiskit") from exc

    qc = QuantumCircuit(n)
    for target in range(n):
        qc.h(target)
        for control in range(target + 1, n):
            qc.cp(pi / 2 ** (control - target), control, target)
    for i in range(n // 2):
        qc.swap(i, n - i - 1)
    return qc


if __name__ == "__main__":
    print(build_qft().draw(output="text"))

