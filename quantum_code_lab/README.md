# Quantum Code Lab

This directory is an execution sandbox containing runnable Python implementations of core quantum computing concepts, hybrid classical-quantum models, and quantum machine learning (QML) experiments. All scripts are optimized to run quickly using local simulations.

---

## 1. Directory Architecture

The code lab is structured around a central command-line interface (`cli.py`) that acts as a unified runner for all modules.

```
quantum_code_lab/
├── README.md                   # This directory guide
├── cli.py                      # Central demo runner
├── basic_quantum_circuits.py   # Qiskit circuit basics
├── bell_states.py              # Qiskit Bell state generator
├── cirq_bell_state.py          # Google Cirq Bell state simulation
├── quantum_fourier_transform.py# Iterative QFT builder
├── grover_search.py            # 2-qubit Grover circuit drawer
├── shor_algorithm_demo.py      # Shor's order finding simulation
├── sympy_quantum_math.py       # Symbolic algebra calculations
├── qml_hybrid_model.py         # PennyLane hybrid quantum neural network
├── vqe_molecule_simulation.py  # Variational Quantum Eigensolver
├── tensorflow_classifier.py    # Classical Keras neural network reference
└── tensorflow_quantum_pqc.py   # TFQ parameterized circuit prediction
```

---

## 2. Interactive CLI Guide

You can run any demo in the sandbox directly using the unified `cli.py` script. The CLI imports the specific script dynamically and executes its main routine.

### Syntax
```bash
python quantum_code_lab/cli.py [demo_name]
```

### Supported Demos

| Demo Name | Script Target | Command Execution |
| :--- | :--- | :--- |
| `basic` | `basic_quantum_circuits.py` | `python quantum_code_lab/cli.py basic` |
| `bell` | `bell_states.py` | `python quantum_code_lab/cli.py bell` |
| `cirq` | `cirq_bell_state.py` | `python quantum_code_lab/cli.py cirq` |
| `qft` | `quantum_fourier_transform.py` | `python quantum_code_lab/cli.py qft` |
| `grover` | `grover_search.py` | `python quantum_code_lab/cli.py grover` |
| `shor` | `shor_algorithm_demo.py` | `python quantum_code_lab/cli.py shor` |
| `qml` | `qml_hybrid_model.py` | `python quantum_code_lab/cli.py qml` |
| `sympy` | `sympy_quantum_math.py` | `python quantum_code_lab/cli.py sympy` |
| `vqe` | `vqe_molecule_simulation.py` | `python quantum_code_lab/cli.py vqe` |
| `tf` | `tensorflow_classifier.py` | `python quantum_code_lab/cli.py tf` |
| `tfq` | `tensorflow_quantum_pqc.py` | `python quantum_code_lab/cli.py tfq` |

---

## 3. Library & Backend Coverage

The code lab is designed to show you how different industry-standard libraries (Qiskit, Cirq, PennyLane, SymPy, and TensorFlow Quantum) handle quantum states.

| Script | Principal Framework | Backend Simulator / Engine |
| :--- | :--- | :--- |
| `basic_quantum_circuits.py` | Qiskit | Local text-based circuit drawer |
| `bell_states.py` | Qiskit | Local text-based circuit drawer |
| `cirq_bell_state.py` | Cirq | `cirq.Simulator` state engine |
| `qml_hybrid_model.py` | PennyLane | PennyLane's default qubit simulator |
| `vqe_molecule_simulation.py` | PennyLane | Autodiff-compatible qubit simulator |
| `tensorflow_quantum_pqc.py` | TensorFlow Quantum | Cirq simulation via TFQ layers |
| `sympy_quantum_math.py` | SymPy | Symbolic engine (algebraic derivation) |

---

## 4. Installation & Environment Setup

To run the examples in this folder, install the package requirements located in the repository root:

```bash
pip install -r requirements.txt
```

### Dependency Stack Specifications
* **Qiskit (>=2.0)**: Used for state preparation and circuit illustrations. All legacy patterns (such as `c_if`) have been replaced with modern Qiskit 2.x conventions.
* **PennyLane (>=0.39)**: Used for variational circuits and differentiable quantum computational models.
* **TensorFlow (2.18.1) & TensorFlow Quantum (0.7.6)**: Requires a matching Python version (usually Python 3.9, 3.10, or 3.11 depending on OS). If TensorFlow Quantum is not installed, the non-TFQ scripts remain fully functional.

---

## 5. Glossary of QML & Simulation Terms

* **Ansatz**: The layout or structure of a parameterized quantum circuit used in variational algorithms. It defines the sequence of entangling gates and parameterized rotations.
* **Parameterized Quantum Circuit (PQC)**: A quantum circuit where some gate parameters (usually rotation angles) are kept variable. It functions as a quantum neural network.
* **Expectation Value**: The average value of a quantum measurement observable. In QML, this serves as the numerical output of a quantum node.
* **Hybrid Classical-Quantum Loop**: An optimization cycle where a quantum computer evaluates expectation values and a classical computer updates gate parameters using gradient-based optimization.
* **Parameter Shift Rule**: An analytical method to evaluate exact gradients of a quantum circuit on physical hardware by running the circuit with shifted parameters ($\theta \pm \pi/2$).
* **Hamiltonian Observable**: A matrix representing the total energy or target state of a system. In VQE and optimization tasks, the Hamiltonian is used to evaluate the loss function.
* **Symbolic Parameter**: A math variable (such as `sympy.Symbol`) used to represent a parameterized gate angle in Cirq or TensorFlow Quantum.
* **Simulator Backend**: A classical software engine that simulates the linear algebra of quantum systems.
