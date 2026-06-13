"""
Quantum Teleportation — Full Protocol Implementation
=====================================================

Demonstrates the quantum teleportation protocol: an unknown qubit state |ψ⟩
is transferred from Alice to Bob using one shared Bell pair and two classical
bits of communication.  The original qubit is destroyed in the process
(consistent with the no-cloning theorem).

Protocol
--------
    1. Alice and Bob share a Bell pair |Φ⁺⟩ = (|00⟩ + |11⟩)/√2.
    2. Alice entangles |ψ⟩ with her half of the Bell pair (CNOT + H).
    3. Alice measures her two qubits, obtaining two classical bits (b₁, b₂).
    4. Bob applies corrections X^{b₂} Z^{b₁} to recover |ψ⟩ on his qubit.

This implementation provides:
    • A deferred-measurement version (full unitary + final measurement)
    • Statevector verification of teleportation fidelity
    • Multiple test states to confirm correctness

Usage
-----
    python implementation.py
"""

from __future__ import annotations

import numpy as np


def _check_qiskit():
    """Ensure Qiskit is installed and return required modules."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, state_fidelity
        return QuantumCircuit, Statevector, state_fidelity
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required.  Install it with:\n"
            "  pip install 'qiskit>=1.0' qiskit-aer"
        ) from exc


# ---------------------------------------------------------------------------
# State preparation helpers
# ---------------------------------------------------------------------------

def prepare_state(theta: float, phi: float) -> "QuantumCircuit":
    """Build a 1-qubit circuit that creates |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩.

    Parameters
    ----------
    theta : float
        Polar angle on the Bloch sphere (0 ≤ θ ≤ π).
    phi : float
        Azimuthal angle on the Bloch sphere (0 ≤ φ < 2π).
    """
    QuantumCircuit, _, _ = _check_qiskit()
    qc = QuantumCircuit(1, name="ψ-prep")
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    return qc


# ---------------------------------------------------------------------------
# Teleportation circuit (deferred measurement)
# ---------------------------------------------------------------------------

def build_teleportation_circuit(
    theta: float = np.pi / 3,
    phi: float = np.pi / 4,
) -> "QuantumCircuit":
    """Build the full quantum teleportation circuit.

    Uses the *deferred measurement* pattern: instead of mid-circuit
    measurement + classical feed-forward, the corrections are applied as
    controlled-X and controlled-Z gates.  The final state of qubit 2 (Bob)
    is exactly |ψ⟩ regardless of Alice's measurement outcomes.

    Qubit layout
    ------------
    q0: Alice's data qubit (holds |ψ⟩)
    q1: Alice's half of the Bell pair
    q2: Bob's half of the Bell pair (receives |ψ⟩)

    Parameters
    ----------
    theta, phi : float
        Bloch-sphere angles defining the state to teleport.
    """
    QuantumCircuit, _, _ = _check_qiskit()

    qc = QuantumCircuit(3, name="Teleportation")

    # --- Step 1: Prepare |ψ⟩ on q0 ---
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.barrier(label="state prepared")

    # --- Step 2: Create Bell pair between q1 (Alice) and q2 (Bob) ---
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier(label="Bell pair")

    # --- Step 3: Alice's Bell-basis measurement (entangle q0 with q1) ---
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier(label="Bell measurement")

    # --- Step 4: Bob's corrections (deferred measurement version) ---
    # CX conditioned on q1 → applies X to q2 when q1=|1⟩
    qc.cx(1, 2)
    # CZ conditioned on q0 → applies Z to q2 when q0=|1⟩
    qc.cz(0, 2)
    qc.barrier(label="corrections")

    return qc


def build_measurement_circuit(
    theta: float = np.pi / 3,
    phi: float = np.pi / 4,
) -> "QuantumCircuit":
    """Build the teleportation circuit with final measurements on all qubits.

    Returns
    -------
    QuantumCircuit
        3-qubit, 3-classical-bit circuit ready for sampling.
    """
    QuantumCircuit, _, _ = _check_qiskit()

    teleport = build_teleportation_circuit(theta, phi)
    qc = QuantumCircuit(3, 3, name="Teleport-Measure")
    qc.compose(teleport, inplace=True)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_teleportation(theta: float, phi: float) -> float:
    """Verify that Bob's qubit ends up in the correct state |ψ⟩.

    Computes the *state fidelity* between the ideal |ψ⟩ and the reduced
    density matrix of Bob's qubit (q2) after the teleportation circuit.

    Returns
    -------
    float
        Fidelity ∈ [0, 1].  A value of 1.0 means perfect teleportation.
    """
    _, Statevector, state_fidelity = _check_qiskit()

    # Ideal target state
    ideal = Statevector.from_instruction(prepare_state(theta, phi))

    # Full 3-qubit statevector after teleportation
    teleport_circ = build_teleportation_circuit(theta, phi)
    full_sv = Statevector.from_instruction(teleport_circ)

    # Trace out Alice's qubits (q0, q1) → Bob's reduced density matrix
    rho_bob = full_sv.trace([0, 1])

    # Fidelity between ideal pure state and Bob's reduced state
    fidelity = state_fidelity(ideal, rho_bob)
    return fidelity


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full teleportation demonstration."""
    print("=" * 65)
    print("  QUANTUM TELEPORTATION — PROTOCOL DEMONSTRATION")
    print("=" * 65)

    # Test cases: various points on the Bloch sphere
    test_states = [
        ("∣0⟩",           0.0,          0.0),
        ("∣1⟩",           np.pi,        0.0),
        ("∣+⟩",           np.pi / 2,    0.0),
        ("∣−⟩",           np.pi / 2,    np.pi),
        ("∣+i⟩",          np.pi / 2,    np.pi / 2),
        ("arbitrary ψ₁",  np.pi / 3,    np.pi / 4),
        ("arbitrary ψ₂",  2 * np.pi / 5, 3 * np.pi / 7),
    ]

    # --- 1. Circuit diagram (for the default state) ---
    print("\n▸ TELEPORTATION CIRCUIT (θ=π/3, φ=π/4)\n")
    qc = build_teleportation_circuit()
    print(qc.draw(output="text"))
    print()

    # --- 2. Fidelity verification for each test state ---
    print("▸ FIDELITY VERIFICATION\n")
    print(f"  {'State':<18s} {'θ':>8s} {'φ':>8s} {'Fidelity':>10s}  {'Status'}")
    print("  " + "─" * 55)

    all_pass = True
    for label, theta, phi in test_states:
        fid = verify_teleportation(theta, phi)
        status = "✓ PASS" if abs(fid - 1.0) < 1e-10 else "✗ FAIL"
        if "FAIL" in status:
            all_pass = False
        print(f"  {label:<18s} {theta:8.4f} {phi:8.4f} {fid:10.6f}  {status}")

    print()
    if all_pass:
        print("  ★ All test states teleported with perfect fidelity (F = 1.0).")
    else:
        print("  ✗ Some states failed — check the output above.")

    # --- 3. Measurement statistics ---
    _, Statevector, _ = _check_qiskit()
    SHOTS = 8192
    print(f"\n▸ MEASUREMENT STATISTICS ({SHOTS:,} shots, arbitrary state)\n")
    qc_meas = build_measurement_circuit(np.pi / 3, np.pi / 4)
    sv = Statevector.from_instruction(
        build_teleportation_circuit(np.pi / 3, np.pi / 4)
    )
    counts = sv.sample_counts(SHOTS)
    for bitstring, count in sorted(counts.items()):
        bar = "█" * int(40 * count / SHOTS)
        print(f"  |{bitstring}⟩  {count:5d}  ({count/SHOTS:6.2%})  {bar}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
