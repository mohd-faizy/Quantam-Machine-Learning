"""Bernstein-Vazirani hidden string demo."""


def build_circuit(secret: str = "1011"):
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise SystemExit("Install qiskit to run this demo: pip install qiskit") from exc

    n = len(secret)
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    for i, bit in enumerate(reversed(secret)):
        if bit == "1":
            qc.cx(i, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


if __name__ == "__main__":
    print(build_circuit().draw(output="text"))

