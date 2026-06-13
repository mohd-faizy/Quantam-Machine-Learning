"""
Quantum Fourier Transform (QFT) — Complete Implementation
==========================================================

The Quantum Fourier Transform is the quantum analogue of the Discrete
Fourier Transform, mapping computational-basis amplitudes into phase-encoded
frequency information.  It is the key subroutine in Shor's algorithm,
quantum phase estimation, and many hidden-structure algorithms.

Features
--------
    • Forward QFT and Inverse QFT circuit builders
    • Configurable number of qubits
    • Round-trip verification (QFT → IQFT = Identity)
    • Demonstration of phase encoding on specific input states
    • Statevector analysis showing frequency representation

Mathematical definition
-----------------------
    QFT|x⟩ = (1/√N) Σ_{y=0}^{N-1} e^{2πixy/N} |y⟩

    where N = 2ⁿ.

Usage
-----
    python implementation.py
"""

from __future__ import annotations

import numpy as np
from math import pi


def _check_qiskit():
    """Ensure Qiskit is installed and return required modules."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, Operator
        return QuantumCircuit, Statevector, Operator
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required.  Install it with:\n"
            "  pip install 'qiskit>=1.0' qiskit-aer"
        ) from exc


# ---------------------------------------------------------------------------
# QFT circuit builders
# ---------------------------------------------------------------------------

def build_qft(n: int = 3, swap: bool = True) -> "QuantumCircuit":
    """Build the Quantum Fourier Transform circuit on n qubits.

    Parameters
    ----------
    n : int
        Number of qubits.
    swap : bool
        Whether to include final SWAP gates for correct bit ordering.

    Returns
    -------
    QuantumCircuit
        The QFT circuit (no measurements).

    Circuit structure
    -----------------
    For each qubit j (from 0 to n−1):
        1. Apply Hadamard to qubit j.
        2. Apply controlled-R_k rotation from qubit k to qubit j,
           for k = j+1, j+2, …, n−1, where R_k = diag(1, e^{2πi/2^k}).
    Finally, reverse qubit order with SWAPs.
    """
    QuantumCircuit, _, _ = _check_qiskit()

    qc = QuantumCircuit(n, name=f"QFT({n})")

    for target in range(n):
        # Hadamard on target qubit
        qc.h(target)

        # Controlled phase rotations
        for control in range(target + 1, n):
            k = control - target
            angle = pi / (2 ** k)
            qc.cp(angle, control, target)

    # Reverse qubit order
    if swap:
        for i in range(n // 2):
            qc.swap(i, n - i - 1)

    return qc


def build_inverse_qft(n: int = 3, swap: bool = True) -> "QuantumCircuit":
    """Build the Inverse Quantum Fourier Transform circuit.

    Equivalent to QFT†: reverses the Fourier mapping.

    Parameters
    ----------
    n : int
        Number of qubits.
    swap : bool
        Whether to include initial SWAP gates (inverse of QFT's final SWAPs).
    """
    QuantumCircuit, _, _ = _check_qiskit()

    qft = build_qft(n, swap=swap)
    iqft = qft.inverse()
    iqft.name = f"IQFT({n})"
    return iqft


def build_qft_with_input(n: int, input_state: int) -> "QuantumCircuit":
    """Build a circuit that prepares |input_state⟩ then applies QFT.

    Parameters
    ----------
    n : int
        Number of qubits.
    input_state : int
        The computational basis state to Fourier-transform (0 ≤ input_state < 2ⁿ).
    """
    QuantumCircuit, _, _ = _check_qiskit()

    qc = QuantumCircuit(n, name=f"QFT(|{input_state}⟩)")

    # Prepare the input state |input_state⟩
    binary = format(input_state, f"0{n}b")
    for i, bit in enumerate(reversed(binary)):
        if bit == "1":
            qc.x(i)

    qc.barrier(label="input prepared")

    # Apply QFT
    qft = build_qft(n)
    qc.compose(qft, inplace=True)

    return qc


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_round_trip(n: int = 3) -> bool:
    """Verify that QFT followed by IQFT equals the identity.

    Tests on multiple computational basis states.
    """
    QuantumCircuit, Statevector, _ = _check_qiskit()

    qft = build_qft(n)
    iqft = build_inverse_qft(n)

    all_pass = True
    for state_int in range(2 ** n):
        # Prepare |state⟩
        qc = QuantumCircuit(n)
        binary = format(state_int, f"0{n}b")
        for i, bit in enumerate(reversed(binary)):
            if bit == "1":
                qc.x(i)

        # Apply QFT then IQFT
        qc.compose(qft, inplace=True)
        qc.compose(iqft, inplace=True)

        sv = Statevector.from_instruction(qc)

        # Check that we get back |state⟩
        expected = np.zeros(2 ** n, dtype=complex)
        expected[state_int] = 1.0

        if not np.allclose(sv.data, expected, atol=1e-10):
            print(f"  ✗ Round-trip failed for |{binary}⟩")
            all_pass = False

    return all_pass


def verify_qft_matrix(n: int = 3) -> bool:
    """Verify that the QFT circuit produces the correct unitary matrix.

    Compares the circuit's unitary against the analytically computed DFT matrix.
    """
    _, _, Operator = _check_qiskit()

    qft = build_qft(n)
    N = 2 ** n

    # Extract unitary from circuit
    circuit_unitary = Operator(qft).data

    # Build the expected DFT matrix
    omega = np.exp(2j * np.pi / N)
    expected = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            expected[i, j] = omega ** (i * j) / np.sqrt(N)

    return np.allclose(circuit_unitary, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_qft_output(n: int, input_state: int) -> dict:
    """Analyze the QFT output for a given input state.

    Returns
    -------
    dict
        Mapping from basis state label to (amplitude, probability, phase).
    """
    _, Statevector, _ = _check_qiskit()

    qc = build_qft_with_input(n, input_state)
    sv = Statevector.from_instruction(qc)

    results = {}
    for i in range(2 ** n):
        amp = sv.data[i]
        prob = abs(amp) ** 2
        phase = np.angle(amp) if abs(amp) > 1e-10 else 0.0
        label = format(i, f"0{n}b")
        results[label] = (amp, prob, phase)

    return results


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full QFT demonstration."""
    print("=" * 70)
    print("  QUANTUM FOURIER TRANSFORM — DEMONSTRATION")
    print("=" * 70)

    # --- 1. Circuit diagrams ---
    for n in [3, 4]:
        print(f"\n▸ QFT CIRCUIT ({n} qubits)\n")
        qft = build_qft(n)
        print(qft.draw(output="text"))
        print()

    print(f"▸ INVERSE QFT CIRCUIT (3 qubits)\n")
    iqft = build_inverse_qft(3)
    print(iqft.draw(output="text"))
    print()

    # --- 2. Round-trip verification ---
    print("▸ ROUND-TRIP VERIFICATION (QFT → IQFT = I)\n")
    for n in [2, 3, 4]:
        ok = verify_round_trip(n)
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {n}-qubit round-trip: {status}  "
              f"(tested all {2**n} basis states)")

    # --- 3. Unitary matrix verification ---
    print("\n▸ UNITARY MATRIX VERIFICATION\n")
    for n in [2, 3, 4]:
        ok = verify_qft_matrix(n)
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {n}-qubit QFT ≡ DFT matrix: {status}")

    # --- 4. QFT output analysis ---
    n = 3
    N = 2 ** n
    print(f"\n▸ QFT OUTPUT ANALYSIS ({n} qubits, N={N})\n")

    for input_state in [0, 1, 3, 5]:
        print(f"  Input: |{format(input_state, f'0{n}b')}⟩  (decimal {input_state})")
        results = analyze_qft_output(n, input_state)
        print(f"  {'State':<8s} {'Probability':>12s} {'Phase (rad)':>12s} {'Phase (π)':>10s}")
        print(f"  {'─' * 45}")
        for label, (amp, prob, phase) in results.items():
            if prob > 1e-10:
                phase_pi = phase / np.pi if abs(phase) > 1e-10 else 0.0
                bar = "█" * int(20 * prob)
                print(f"  |{label}⟩ {prob:12.6f} {phase:12.6f} {phase_pi:9.4f}π  {bar}")
        print()

    # --- 5. Measurement statistics ---
    _, Statevector, _ = _check_qiskit()
    SHOTS = 8192
    print(f"▸ MEASUREMENT SAMPLING ({SHOTS:,} shots)\n")

    for input_state in [0, 3]:
        print(f"  QFT|{format(input_state, f'03b')}⟩:")
        qc = build_qft_with_input(3, input_state)
        sv = Statevector.from_instruction(qc)
        counts = sv.sample_counts(SHOTS)
        for bitstring, count in sorted(counts.items()):
            bar = "█" * int(40 * count / SHOTS)
            print(f"    |{bitstring}⟩  {count:5d}  ({count/SHOTS:6.2%})  {bar}")
        print()

    print("=" * 70)
    print("  ★ QFT verified: O(n²) gates for an exponentially large transform.")
    print("=" * 70)


if __name__ == "__main__":
    main()
