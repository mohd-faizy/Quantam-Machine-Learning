"""Tiny VQE-style optimization for a single-qubit toy Hamiltonian."""

from __future__ import annotations


def run_vqe(steps: int = 40, learning_rate: float = 0.15) -> tuple[float, float]:
    try:
        import pennylane as qml
        from pennylane import numpy as np
    except ImportError as exc:
        raise SystemExit("Install PennyLane: pip install pennylane") from exc

    dev = qml.device("default.qubit", wires=1)
    hamiltonian = qml.Hamiltonian([1.0, -0.5], [qml.PauliZ(0), qml.PauliX(0)])

    @qml.qnode(dev)
    def energy(theta):
        qml.RY(theta, wires=0)
        return qml.expval(hamiltonian)

    theta = np.array(0.1, requires_grad=True)
    opt = qml.GradientDescentOptimizer(learning_rate)
    for _ in range(steps):
        theta = opt.step(energy, theta)
    return float(theta), float(energy(theta))


if __name__ == "__main__":
    angle, value = run_vqe()
    print(f"theta={angle:.6f}, energy={value:.6f}")

