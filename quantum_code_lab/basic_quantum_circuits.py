"""Basic one- and two-qubit circuits."""


def build_circuits():
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit("Install qiskit: pip install qiskit") from exc

    hadamard = QuantumCircuit(1, 1)
    hadamard.h(0)
    hadamard.measure(0, 0)

    x_gate = QuantumCircuit(1, 1)
    x_gate.x(0)
    x_gate.measure(0, 0)

    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])
    return {"hadamard": hadamard, "x_gate": x_gate, "bell": bell}


if __name__ == "__main__":
    for name, circuit in build_circuits().items():
        print(f"\n{name}\n{circuit.draw(output='text')}")

