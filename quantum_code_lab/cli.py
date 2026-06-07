"""Small CLI for running educational demos."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quantum learning demo.")
    parser.add_argument(
        "demo",
        choices=["basic", "bell", "cirq", "qft", "grover", "shor", "qml", "tf", "tfq", "sympy", "vqe"],
    )
    args = parser.parse_args()

    if args.demo == "basic":
        from basic_quantum_circuits import build_circuits

        for name, circuit in build_circuits().items():
            print(f"\n{name}\n{circuit.draw(output='text')}")
    elif args.demo == "bell":
        from bell_states import bell_state

        print(bell_state().draw(output="text"))
    elif args.demo == "cirq":
        from cirq_bell_state import run_bell_state

        circuit, counts = run_bell_state()
        print(circuit)
        print(counts)
    elif args.demo == "qft":
        from quantum_fourier_transform import qft

        print(qft(3).draw(output="text"))
    elif args.demo == "grover":
        from grover_search import grover_circuit

        print(grover_circuit().draw(output="text"))
    elif args.demo == "shor":
        from shor_algorithm_demo import shor_demo

        print(shor_demo())
    elif args.demo == "qml":
        from qml_hybrid_model import pennylane_vqc_prediction

        print(pennylane_vqc_prediction())
    elif args.demo == "tf":
        from tensorflow_classifier import train_tiny_classifier

        print(train_tiny_classifier())
    elif args.demo == "tfq":
        from tensorflow_quantum_pqc import build_tfq_pqc_prediction

        print(build_tfq_pqc_prediction())
    elif args.demo == "sympy":
        from sympy_quantum_math import grover_rotation_formula

        print(grover_rotation_formula())
    elif args.demo == "vqe":
        from vqe_molecule_simulation import run_vqe

        print(run_vqe())


if __name__ == "__main__":
    main()
