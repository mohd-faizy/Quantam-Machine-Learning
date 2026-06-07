"""Create the four Bell states."""


def bell_state(label: str = "phi_plus"):
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit("Install qiskit: pip install qiskit") from exc

    qc = QuantumCircuit(2, 2)
    if label in {"psi_plus", "psi_minus"}:
        qc.x(1)
    qc.h(0)
    qc.cx(0, 1)
    if label in {"phi_minus", "psi_minus"}:
        qc.z(0)
    qc.measure([0, 1], [0, 1])
    return qc


if __name__ == "__main__":
    for state in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
        print(f"\n{state}\n{bell_state(state).draw(output='text')}")

