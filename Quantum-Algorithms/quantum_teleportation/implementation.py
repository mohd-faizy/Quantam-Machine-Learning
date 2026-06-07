"""Quantum teleportation circuit."""


def build_circuit():
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit("Install qiskit to run this demo: pip install qiskit") from exc

    qc = QuantumCircuit(3, 2)
    qc.ry(0.7, 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    qc.x(2).c_if(qc.clbits[1], 1)
    qc.z(2).c_if(qc.clbits[0], 1)
    return qc


if __name__ == "__main__":
    print(build_circuit().draw(output="text"))

