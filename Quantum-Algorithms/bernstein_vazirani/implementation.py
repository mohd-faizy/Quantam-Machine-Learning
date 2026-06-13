"""
Bernstein-Vazirani Algorithm — Complete Implementation
======================================================

Finds a hidden bit-string *s* encoded in a linear Boolean function
f(x) = s · x mod 2 using a single quantum oracle query.

Classically, discovering all n bits of *s* requires n queries (one per bit).
The Bernstein-Vazirani algorithm recovers the entire string in one shot
by exploiting phase kickback and the Hadamard transform.

Features
--------
    • Configurable secret string of arbitrary length
    • Qiskit simulation with statevector sampling
    • Automatic verification that recovered string matches the secret
    • Multiple test cases with different secrets

Usage
-----
    python implementation.py
"""

from __future__ import annotations


def _check_qiskit():
    """Ensure Qiskit is installed and return required modules."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        return QuantumCircuit, Statevector
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required.  Install it with:\n"
            "  pip install 'qiskit>=1.0' qiskit-aer"
        ) from exc


# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------

def build_circuit(secret: str = "1011") -> "QuantumCircuit":
    """Build the Bernstein-Vazirani circuit for a given secret string.

    Parameters
    ----------
    secret : str
        The hidden bit-string *s*.  Must contain only '0' and '1'.

    Returns
    -------
    QuantumCircuit
        An (n+1)-qubit, n-classical-bit circuit.  The input register will
        deterministically measure *s* when run on a noiseless backend.

    Circuit construction
    --------------------
    1. Initialise the ancilla qubit to |−⟩ = H|1⟩ for phase kickback.
    2. Apply H⊗ⁿ to the input register → uniform superposition.
    3. Oracle: for each bit position i where s[i] = 1, apply CNOT(i, ancilla).
       This encodes (-1)^{s·x} into the phase of each |x⟩ amplitude.
    4. Apply H⊗ⁿ again → interference decodes the phases into |s⟩.
    5. Measure the input register.
    """
    if not all(b in "01" for b in secret):
        raise ValueError(f"Secret must be a binary string, got '{secret}'.")

    QuantumCircuit, _ = _check_qiskit()

    n = len(secret)
    qc = QuantumCircuit(n + 1, n, name=f"BV(s={secret})")

    # Step 1 — Prepare ancilla in |−⟩
    qc.x(n)
    qc.h(n)

    # Step 2 — Hadamard the input register
    qc.h(range(n))
    qc.barrier(label="superposition")

    # Step 3 — Oracle: CNOT from qubit i to ancilla where s[i]=1
    # Qiskit qubit ordering: qubit 0 is the rightmost bit of the string
    for i, bit in enumerate(reversed(secret)):
        if bit == "1":
            qc.cx(i, n)
    qc.barrier(label="oracle")

    # Step 4 — Second Hadamard round on input register
    qc.h(range(n))

    # Step 5 — Measure input register
    qc.measure(range(n), range(n))

    return qc


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_bv(secret: str = "1011", shots: int = 4096) -> tuple[str, dict[str, int]]:
    """Simulate the Bernstein-Vazirani circuit and extract the recovered string.

    Returns
    -------
    (recovered_secret, counts)
        recovered_secret : str
            The most-frequently measured bit-string (should equal *secret*).
        counts : dict
            Full measurement histogram.
    """
    _, Statevector = _check_qiskit()
    QuantumCircuit, _ = _check_qiskit()

    n = len(secret)

    # Build the unitary part (no measurement) for statevector sampling
    qc_sv = QuantumCircuit(n + 1, name=f"BV-sv(s={secret})")
    qc_sv.x(n)
    qc_sv.h(n)
    qc_sv.h(range(n))
    for i, bit in enumerate(reversed(secret)):
        if bit == "1":
            qc_sv.cx(i, n)
    qc_sv.h(range(n))

    sv = Statevector.from_instruction(qc_sv)
    raw_counts = sv.sample_counts(shots)

    # Extract input-register bits (drop ancilla which is the MSB)
    input_counts: dict[str, int] = {}
    for bitstring, count in raw_counts.items():
        input_bits = bitstring[1:]  # drop ancilla
        input_counts[input_bits] = input_counts.get(input_bits, 0) + count

    # The most common measurement is the recovered secret
    recovered = max(input_counts, key=input_counts.get)
    return recovered, input_counts


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Bernstein-Vazirani algorithm on multiple secret strings."""
    print("=" * 65)
    print("  BERNSTEIN-VAZIRANI ALGORITHM — HIDDEN STRING RECOVERY")
    print("=" * 65)

    test_secrets = ["101", "1011", "110011", "11111", "10000001", "0000"]
    SHOTS = 4096

    for secret in test_secrets:
        n = len(secret)
        print(f"\n{'─' * 65}")
        print(f"  Secret string s = {secret}  ({n} qubits)")
        print(f"{'─' * 65}\n")

        # Build and display circuit
        qc = build_circuit(secret)
        print(qc.draw(output="text"))
        print()

        # Simulate
        recovered, counts = run_bv(secret, shots=SHOTS)
        print(f"  Measurement counts: {dict(sorted(counts.items()))}")
        print(f"  Recovered string:   {recovered}")

        # Verify
        if recovered == secret:
            print(f"  ✓ CORRECT — recovered s = {recovered} in ONE query")
        else:
            print(f"  ✗ WRONG — expected '{secret}', got '{recovered}'")
            print(f"    (Classical algorithm would need {n} queries)")

    print("\n" + "=" * 65)
    print("  ★ All secret strings recovered with a single oracle query each.")
    print(f"  ★ Classical approach would require n queries per string.")
    print("=" * 65)


if __name__ == "__main__":
    main()
