# Quantum Code Lab

Runnable examples for learning quantum circuits and QML. The scripts prefer small circuits so they can run locally and in CI.

## Examples

```bash
python quantum_code_lab/basic_quantum_circuits.py
python quantum_code_lab/bell_states.py
python quantum_code_lab/cirq_bell_state.py
python quantum_code_lab/quantum_fourier_transform.py
python quantum_code_lab/grover_search.py
python quantum_code_lab/shor_algorithm_demo.py
python quantum_code_lab/qml_hybrid_model.py
python quantum_code_lab/tensorflow_classifier.py
python quantum_code_lab/tensorflow_quantum_pqc.py
python quantum_code_lab/sympy_quantum_math.py
python quantum_code_lab/vqe_molecule_simulation.py
python quantum_code_lab/cli.py bell
```

## Install

```bash
pip install -r requirements.txt
```

TensorFlow Quantum currently expects a compatible Python/TensorFlow pairing. This repository pins `tensorflow==2.18.1` and `tensorflow-quantum==0.7.6` in the unified requirements file.

## Backend Coverage

| File | Backend |
|---|---|
| `basic_quantum_circuits.py` | Qiskit |
| `bell_states.py` | Qiskit |
| `cirq_bell_state.py` | Cirq |
| `qml_hybrid_model.py` | PennyLane plus TFQ summary |
| `tensorflow_classifier.py` | TensorFlow/Keras |
| `tensorflow_quantum_pqc.py` | TensorFlow Quantum, Cirq, SymPy |
| `sympy_quantum_math.py` | SymPy symbolic derivation |
| `vqe_molecule_simulation.py` | PennyLane |

