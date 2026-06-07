"""Deutsch-Jozsa demo with a balanced parity oracle."""


def build_circuit(n: int = 3):
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit("Install qiskit to run this demo: pip install qiskit") from exc

    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    for qubit in range(n):
        qc.cx(qubit, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


if __name__ == "__main__":
    print(build_circuit().draw(output="text"))

