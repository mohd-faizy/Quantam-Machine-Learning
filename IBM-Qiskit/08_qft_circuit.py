"""
08 — Quantum Fourier Transform (QFT)
=====================================
Build the standard QFT circuit for n qubits using Hadamard gates,
controlled-phase rotations, and final SWAP gates.

The QFT is the quantum analogue of the discrete Fourier transform and
is a key building block in Shor's algorithm, quantum phase estimation,
and other frequency-domain algorithms.

Run:
    python IBM-Qiskit/08_qft_circuit.py
"""

from __future__ import annotations

from math import pi


def qft_circuit(n: int = 4):
    """Build an n-qubit QFT circuit.

    Args:
        n: number of qubits (default 4).
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required. Install it with:  pip install qiskit"
        ) from exc

    qc = QuantumCircuit(n, name="QFT")

    for target in range(n):
        qc.h(target)
        # Controlled-phase rotations from higher qubits
        for control in range(target + 1, n):
            angle = pi / 2 ** (control - target)
            qc.cp(angle, control, target)
        qc.barrier()

    # Reverse qubit order (bit-reversal step)
    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    return qc


def inverse_qft_circuit(n: int = 4):
    """Build the inverse QFT by inverting the QFT circuit."""
    return qft_circuit(n).inverse()


def main() -> None:
    for n in [3, 4]:
        qc = qft_circuit(n)
        print(f"\n{'═' * 50}")
        print(f"  {n}-qubit Quantum Fourier Transform")
        print(f"{'═' * 50}")
        print(qc.draw(output="text"))

    # Also show the inverse
    inv = inverse_qft_circuit(4)
    print(f"\n{'═' * 50}")
    print("  4-qubit Inverse QFT")
    print(f"{'═' * 50}")
    print(inv.draw(output="text"))


if __name__ == "__main__":
    main()
