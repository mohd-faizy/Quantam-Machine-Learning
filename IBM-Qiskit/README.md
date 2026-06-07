# IBM Qiskit — Beginner Learning Track

A progressive, hands-on introduction to quantum computing with **IBM Qiskit** (2.x).
Every script is self-contained — just run it and study the output.

## Prerequisites

```bash
pip install qiskit          # or: pip install -r requirements.txt
```

> **Note:** All scripts use the Qiskit 2.x API. The legacy `c_if` method
> (removed in Qiskit 2.0) is **not** used; conditional operations use the
> modern `if_test` context manager instead.

## Learning Track

| # | Script | Topic | What you learn |
|---|---|---|---|
| 1 | [01_single_qubit_gates.py](01_single_qubit_gates.py) | H, X, Y, Z, S, T gates | Gate fundamentals and text-based circuit drawing |
| 2 | [02_multi_qubit_gates.py](02_multi_qubit_gates.py) | CNOT, CZ, Toffoli, SWAP | Multi-qubit entangling operations |
| 3 | [03_bell_states.py](03_bell_states.py) | All 4 Bell states | Entanglement and measurement correlations |
| 4 | [04_quantum_teleportation.py](04_quantum_teleportation.py) | Teleportation protocol | Classical conditioning with `if_test` (Qiskit 2.x) |
| 5 | [05_bernstein_vazirani.py](05_bernstein_vazirani.py) | Bernstein-Vazirani algorithm | Oracle construction and hidden-string recovery |
| 6 | [06_deutsch_jozsa.py](06_deutsch_jozsa.py) | Deutsch-Jozsa algorithm | Constant vs balanced oracle separation |
| 7 | [07_grover_search.py](07_grover_search.py) | Grover's search (2 qubits) | Oracle, diffusion, amplitude amplification |
| 8 | [08_qft_circuit.py](08_qft_circuit.py) | Quantum Fourier Transform | Controlled-phase rotations and bit reversal |

## Quick Start

```bash
# Run any script directly
python IBM-Qiskit/01_single_qubit_gates.py
python IBM-Qiskit/03_bell_states.py
python IBM-Qiskit/07_grover_search.py
```

## Suggested Reading Order

1. **Start with gates** — Scripts 01 and 02 cover the building blocks.
2. **Explore entanglement** — Script 03 shows how Bell states work.
3. **Apply protocols** — Script 04 demonstrates quantum teleportation.
4. **Learn algorithms** — Scripts 05–08 progressively introduce query
   algorithms and the QFT.

## Qiskit 2.x Compatibility

All code in this folder is written for **Qiskit ≥ 2.0** (released March 2025).
Key API patterns used:

- `QuantumCircuit.if_test()` for classical conditioning (replaces `c_if`).
- `qiskit.quantum_info.Statevector` for local simulation.
- Standard gate methods: `.h()`, `.x()`, `.cx()`, `.cz()`, `.cp()`, `.swap()`, etc.
