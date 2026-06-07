"""Two-qubit Grover search for marked state |11>."""


def grover_circuit():
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit("Install qiskit: pip install qiskit") from exc

    qc = QuantumCircuit(2, 2)
    qc.h([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    qc.z([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1])
    qc.measure([0, 1], [0, 1])
    return qc


if __name__ == "__main__":
    print(grover_circuit().draw(output="text"))

