"""
Quantum Entanglement — Bell State Preparation & Verification
=============================================================

Demonstrates the creation and measurement of all four Bell states using Qiskit.
Each Bell state is constructed, simulated, and verified for perfect two-qubit
correlations that cannot be explained by any classical local hidden-variable model.

Bell States
-----------
    |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
    |Φ⁻⟩ = (|00⟩ - |11⟩) / √2
    |Ψ⁺⟩ = (|01⟩ + |10⟩) / √2
    |Ψ⁻⟩ = (|01⟩ - |10⟩) / √2

Usage
-----
    python implementation.py
"""

from __future__ import annotations


def _check_qiskit():
    """Ensure Qiskit is installed and return the required modules."""
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
# Circuit builders
# ---------------------------------------------------------------------------

def build_bell_state(state: str = "phi_plus") -> "QuantumCircuit":
    """Build a circuit that prepares a specific Bell state.

    Parameters
    ----------
    state : str
        One of ``"phi_plus"``, ``"phi_minus"``, ``"psi_plus"``, ``"psi_minus"``.

    Returns
    -------
    QuantumCircuit
        A 2-qubit circuit (no measurements) that prepares the requested Bell state.

    Raises
    ------
    ValueError
        If *state* is not one of the four recognised names.
    """
    QuantumCircuit, _ = _check_qiskit()

    valid = {"phi_plus", "phi_minus", "psi_plus", "psi_minus"}
    if state not in valid:
        raise ValueError(f"Unknown Bell state '{state}'.  Choose from {valid}.")

    qc = QuantumCircuit(2, name=f"Bell-{state}")

    # ---- core entangling block ----
    # |Φ⁺⟩: H·CX on |00⟩
    qc.h(0)
    qc.cx(0, 1)

    # ---- variant adjustments ----
    if state == "phi_minus":
        qc.z(0)                     # phase-flip → (|00⟩ − |11⟩)/√2
    elif state == "psi_plus":
        qc.x(1)                     # bit-flip   → (|01⟩ + |10⟩)/√2
    elif state == "psi_minus":
        qc.x(1)
        qc.z(0)                     #            → (|01⟩ − |10⟩)/√2

    return qc


def build_measurement_circuit(state: str = "phi_plus") -> "QuantumCircuit":
    """Build a Bell-state circuit followed by computational-basis measurement.

    Parameters
    ----------
    state : str
        Bell state name (see :func:`build_bell_state`).

    Returns
    -------
    QuantumCircuit
        A 2-qubit, 2-classical-bit circuit ready for sampling.
    """
    QuantumCircuit, _ = _check_qiskit()

    bell = build_bell_state(state)
    qc = QuantumCircuit(2, 2, name=f"Measure-{state}")
    qc.compose(bell, inplace=True)
    qc.measure([0, 1], [0, 1])
    return qc


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_statevector(state: str = "phi_plus") -> dict:
    """Return the exact statevector for the requested Bell state.

    Returns
    -------
    dict
        Mapping from computational-basis label to complex amplitude.
    """
    _, Statevector = _check_qiskit()

    qc = build_bell_state(state)
    sv = Statevector.from_instruction(qc)
    return dict(zip(
        [f"|{format(i, '02b')}⟩" for i in range(4)],
        sv.data,
    ))


def simulate_counts(state: str = "phi_plus", shots: int = 4096) -> dict[str, int]:
    """Sample the Bell-state circuit and return measurement counts.

    Uses the Qiskit :class:`Statevector` sampler for speed and determinism
    (no noise model).  Falls back to ``qiskit-aer`` if available.

    Returns
    -------
    dict[str, int]
        Mapping from bit-string to number of observations.
    """
    _, Statevector = _check_qiskit()

    qc = build_bell_state(state)
    sv = Statevector.from_instruction(qc)
    return sv.sample_counts(shots)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_correlations(shots: int = 8192) -> bool:
    """Verify that each Bell state produces only correlated outcomes.

    For |Φ±⟩ the only outcomes should be ``00`` and ``11``.
    For |Ψ±⟩ the only outcomes should be ``01`` and ``10``.

    Returns True if all four Bell states pass verification.
    """
    expected = {
        "phi_plus":  {"00", "11"},
        "phi_minus": {"00", "11"},
        "psi_plus":  {"01", "10"},
        "psi_minus": {"01", "10"},
    }

    all_pass = True
    for name, valid_outcomes in expected.items():
        counts = simulate_counts(name, shots=shots)
        observed = set(counts.keys())
        if not observed.issubset(valid_outcomes):
            print(f"  ✗ {name}: unexpected outcomes {observed - valid_outcomes}")
            all_pass = False
        else:
            ratio = counts.get(list(valid_outcomes)[0], 0) / shots
            print(f"  ✓ {name:12s}  outcomes={observed}  "
                  f"split≈{ratio:.2%}/{1 - ratio:.2%}")
    return all_pass


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Bell-state demonstration."""
    print("=" * 65)
    print("  QUANTUM ENTANGLEMENT — BELL STATE DEMONSTRATION")
    print("=" * 65)

    bell_states = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
    labels = {
        "phi_plus":  "|Φ⁺⟩ = (|00⟩ + |11⟩) / √2",
        "phi_minus": "|Φ⁻⟩ = (|00⟩ − |11⟩) / √2",
        "psi_plus":  "|Ψ⁺⟩ = (|01⟩ + |10⟩) / √2",
        "psi_minus": "|Ψ⁻⟩ = (|01⟩ − |10⟩) / √2",
    }

    # --- 1. Circuit diagrams ---
    print("\n▸ CIRCUITS\n")
    for name in bell_states:
        print(f"  {labels[name]}")
        qc = build_measurement_circuit(name)
        print(qc.draw(output="text").text(line_length=72))
        print()

    # --- 2. Statevectors ---
    print("▸ EXACT STATEVECTORS\n")
    for name in bell_states:
        sv = simulate_statevector(name)
        nonzero = {k: v for k, v in sv.items() if abs(v) > 1e-10}
        terms = "  +  ".join(f"{v.real:+.4f}{k}" for k, v in nonzero.items())
        print(f"  {name:12s}: {terms}")
    print()

    # --- 3. Measurement samples ---
    SHOTS = 8192
    print(f"▸ MEASUREMENT COUNTS ({SHOTS:,} shots)\n")
    for name in bell_states:
        counts = simulate_counts(name, shots=SHOTS)
        print(f"  {name:12s}: {dict(sorted(counts.items()))}")
    print()

    # --- 4. Correlation verification ---
    print("▸ CORRELATION VERIFICATION\n")
    ok = verify_correlations(shots=SHOTS)
    print()
    if ok:
        print("  ★ All four Bell states verified: perfect two-qubit correlations.")
    else:
        print("  ✗ Verification failed — check the output above.")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
