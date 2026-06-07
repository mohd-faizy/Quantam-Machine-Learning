# IBM Qiskit Learning Track

This directory contains a progressive, hands-on series of educational scripts designed to teach gate-model quantum computing using **IBM Qiskit (2.x)**.

---

## 1. Qiskit 2.x Architecture & Compatibility

This learning track is built exclusively for the modern **Qiskit 2.0+ SDK** ecosystem. It avoids legacy practices to ensure that your skills align with the latest industry standards.

### Key API Patterns Used:
* **`QuantumCircuit.if_test()`**: Used for conditional execution (classical conditioning) during circuit runtime. The legacy `.c_if()` method (removed in Qiskit 2.0) has been fully replaced.
* **`qiskit.quantum_info.Statevector`**: Used for portable, exact mathematical simulation of circuit states prior to measurement.
* **`qiskit.primitives.StatevectorSampler`**: Used for sampling bitstrings from quantum state measurements, representing the modern Qiskit Primitives workflow.

---

## 2. Learning Track Overview

Follow the scripts in numerical order to build your quantum circuit skills progressively.

| Track Number | Script Name | Core Concept | Quantum Gates Used | Target Outcome |
| :---: | :--- | :--- | :---: | :--- |
| **01** | [01_single_qubit_gates.py](01_single_qubit_gates.py) | Single-Qubit Basics | $H, X, Y, Z, S, T$ | Understanding states on the Bloch sphere and text drawing. |
| **02** | [02_multi_qubit_gates.py](02_multi_qubit_gates.py) | Entangling Operations | $CX, CZ, CCX, SWAP$ | Managing multi-qubit systems and control actions. |
| **03** | [03_bell_states.py](03_bell_states.py) | Bell State Prep | $H, CX, X, Z$ | Generating maximum entanglement and tracking correlations. |
| **04** | [04_quantum_teleportation.py](04_quantum_teleportation.py) | State Transfer Protocol | $H, CX, X, Z$ + `if_test` | Implementing classical conditioning using Qiskit 2.x. |
| **05** | [05_bernstein_vazirani.py](05_bernstein_vazirani.py) | Hidden Parity Search | $H, CX, X$ | Constructing phase oracles to retrieve hidden bitstrings. |
| **06** | [06_deutsch_jozsa.py](06_deutsch_jozsa.py) | Balanced vs. Constant | $H, CX, X$ | Utilizing phase kickback to classify binary functions. |
| **07** | [07_grover_search.py](07_grover_search.py) | Unstructured Search | $H, CZ, Z$ | Implementing oracle marking and amplitude diffusion. |
| **08** | [08_qft_circuit.py](08_qft_circuit.py) | Fourier Transform | $H, CP, SWAP$ | Building phase-rotation networks and computing inverse QFTs. |

---

## 3. Quick Start

Ensure you have Qiskit installed in your environment:

```bash
pip install qiskit
```

Then run any script directly from the project root:

```bash
python IBM-Qiskit/01_single_qubit_gates.py
python IBM-Qiskit/03_bell_states.py
python IBM-Qiskit/07_grover_search.py
```

---

## 4. Suggested Study Path

```
  Step 1: Single-Qubit Gates (01)
                |
                v
  Step 2: Multi-Qubit Gates (02)
                |
                v
  Step 3: Entanglement & Bell States (03)
                |
                v
  Step 4: Quantum Teleportation (04)
                |
                v
  Step 5: Query Algorithms (05 & 06)
                |
                v
  Step 6: Search & Fourier Analysis (07 & 08)
```

---

## 5. Glossary of Qiskit Terms

* **`QuantumCircuit`**: The core data structure in Qiskit representing a sequence of quantum gates, measurements, and registers.
* **`Statevector`**: A representation of a pure state vector in Hilbert space. In Qiskit, it is used for local mathematical debugging without performing measurements.
* **`StatevectorSampler`**: A modern Qiskit Primitive class used to simulate the execution of a circuit and return sampling counts.
* **`if_test()`**: A Qiskit context manager used to perform operations conditionally depending on the state of classical bits.
* **`Barrier`**: A visual and compiler helper inserted into a circuit to prevent compiler optimization across the barrier and to separate circuit phases.
* **`QuantumRegister`**: A register containing qubits.
* **`ClassicalRegister`**: A register containing classical bits, used to store measurement outcomes.
* **Transpilation**: The process of translating a high-level quantum circuit into a lower-level equivalent circuit optimized for a specific target hardware architecture.
* **`draw()`**: A Qiskit circuit method that outputs graphical representations of circuits (available in text, matplotlib, and LaTeX formats).
