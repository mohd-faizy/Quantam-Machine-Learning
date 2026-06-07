"""Bell state simulation with Cirq."""


def run_bell_state(repetitions: int = 100):
    try:
        import cirq
    except ImportError as exc:
        raise SystemExit("Install the unified stack: pip install -r requirements.txt") from exc

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="bell"),
    )
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=repetitions)
    return circuit, result.histogram(key="bell")


if __name__ == "__main__":
    bell_circuit, counts = run_bell_state()
    print(bell_circuit)
    print(counts)

