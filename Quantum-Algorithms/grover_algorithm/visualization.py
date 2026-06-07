"""Visualize the compact Grover circuit."""

from implementation import build_circuit


def main() -> None:
    circuit = build_circuit()
    print(circuit.draw(output="text"))


if __name__ == "__main__":
    main()

