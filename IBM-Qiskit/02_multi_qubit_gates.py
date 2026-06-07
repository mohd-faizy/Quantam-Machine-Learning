"""
02 — Multi-Qubit Gates
======================
Learn the essential multi-qubit entangling gates: CNOT (CX), CZ,
Toffoli (CCX), and SWAP.  Each gate is built in its own small circuit
so you can see exactly what it does.

Run:
    python IBM-Qiskit/02_multi_qubit_gates.py
"""

from __future__ import annotations


def multi_qubit_demos() -> dict:
    """Return a dict of gate-name → QuantumCircuit for multi-qubit gates."""
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required. Install it with:  pip install qiskit"
        ) from exc

    # --- CNOT (CX) ---
    cnot = QuantumCircuit(2, 2)
    cnot.h(0)              # put control in superposition
    cnot.cx(0, 1)          # entangle
    cnot.measure([0, 1], [0, 1])

    # --- CZ ---
    cz = QuantumCircuit(2, 2)
    cz.h(0)
    cz.h(1)
    cz.cz(0, 1)           # controlled-Z
    cz.h(1)               # convert phase to bit-flip for measurement
    cz.measure([0, 1], [0, 1])

    # --- Toffoli (CCX) ---
    toffoli = QuantumCircuit(3, 3)
    toffoli.x(0)           # set control-0 to |1⟩
    toffoli.x(1)           # set control-1 to |1⟩
    toffoli.ccx(0, 1, 2)   # target flips only when both controls are |1⟩
    toffoli.measure([0, 1, 2], [0, 1, 2])

    # --- SWAP ---
    swap = QuantumCircuit(2, 2)
    swap.x(0)              # q0 = |1⟩, q1 = |0⟩
    swap.swap(0, 1)        # after swap: q0 = |0⟩, q1 = |1⟩
    swap.measure([0, 1], [0, 1])

    return {
        "CNOT (CX)": cnot,
        "CZ": cz,
        "Toffoli (CCX)": toffoli,
        "SWAP": swap,
    }


def main() -> None:
    circuits = multi_qubit_demos()
    for name, qc in circuits.items():
        print(f"\n{'─' * 40}")
        print(f"  {name}")
        print(f"{'─' * 40}")
        print(qc.draw(output="text"))


if __name__ == "__main__":
    main()
