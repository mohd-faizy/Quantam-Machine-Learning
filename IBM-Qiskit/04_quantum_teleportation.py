"""
04 — Quantum Teleportation
==========================
Teleport an arbitrary single-qubit state from Alice to Bob using
a shared Bell pair and two classical bits of communication.

This script uses the Qiskit 2.x ``if_test`` context manager
(the legacy ``c_if`` was removed in Qiskit 2.0).

Protocol:
    1. Prepare the state to teleport on q0 (Alice's data qubit).
    2. Create a Bell pair between q1 (Alice's half) and q2 (Bob's half).
    3. Alice performs a Bell measurement on (q0, q1).
    4. Based on the classical results, Bob applies corrections to q2.

Run:
    python IBM-Qiskit/04_quantum_teleportation.py
"""

from __future__ import annotations


def teleportation_circuit(theta: float = 0.7):
    """Build a teleportation circuit that transfers RY(theta)|0⟩.

    Args:
        theta: rotation angle for the state to teleport.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required. Install it with:  pip install qiskit"
        ) from exc

    qc = QuantumCircuit(3, 2)

    # Step 1 — Prepare the state to teleport on q0
    qc.ry(theta, 0)
    qc.barrier()

    # Step 2 — Create Bell pair (q1, q2)
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()

    # Step 3 — Alice's Bell measurement
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    qc.barrier()

    # Step 4 — Bob's conditional corrections (Qiskit 2.x syntax)
    with qc.if_test((qc.clbits[1], 1)):
        qc.x(2)
    with qc.if_test((qc.clbits[0], 1)):
        qc.z(2)

    return qc


def main() -> None:
    qc = teleportation_circuit()
    print("Quantum Teleportation Circuit")
    print("=" * 45)
    print(qc.draw(output="text"))
    print()
    print("After measurement, Bob's qubit q2 holds the")
    print("teleported state RY(0.7)|0⟩, up to Pauli corrections.")


if __name__ == "__main__":
    main()
