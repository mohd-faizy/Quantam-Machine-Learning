# Quantum Algorithms Reference Handbook

This directory serves as the primary theoretical and mathematical hub for the core quantum algorithms implemented in this repository. 

Each algorithm subdirectory contains self-contained learning materials structured as follows:
* `README.md`: High-level intuition, workflow steps, complexity, and limitations.
* `theory.md`: Detailed conceptual explanations and proofs.
* `math.md`: Rigorous mathematical derivations using linear algebra and Bra-Ket notation.
* `implementation.py`: A runnable Python implementation using Qiskit.
* `visualization.py`: Helper scripts to plot measurement histograms and state vectors.

---

## 1. Quick Navigation

| Algorithm/Resource | Main Computational Idea | Directory Path |
| :--- | :--- | :---: |
| **Entanglement** | Harnessing non-classical state correlations as a computational resource. | [entanglement](entanglement/README.md) |
| **Quantum Teleportation** | State transfer via entanglement and two classical bits. | [quantum_teleportation](quantum_teleportation/README.md) |
| **Deutsch-Jozsa** | Determining constant vs. balanced properties of a function in a single query. | [deutsch_jozsa](deutsch_jozsa/README.md) |
| **Bernstein-Vazirani** | Finding a hidden bitstring $s$ in a single query. | [bernstein_vazirani](bernstein_vazirani/README.md) |
| **Simon's Algorithm** | Unveiling a hidden XOR mask with exponential query speedup. | [simon_algorithm](simon_algorithm/README.md) |
| **Grover's Search** | Quadratic speedup for unstructured database searching. | [grover_algorithm](grover_algorithm/README.md) |
| **Quantum Fourier Transform** | Translating state structure from the computational to the phase basis. | [quantum_fourier_transform](quantum_fourier_transform/README.md) |
| **Shor's Algorithm** | Superpolynomial factoring of composite numbers via period finding. | [shor_algorithm](shor_algorithm/README.md) |

---

## 2. Quantum Algorithm Cheat Sheet

| Algorithm | Problem Solved | Speedup | Core Trick / Mechanic | Main Bottleneck |
| :--- | :--- | :---: | :--- | :--- |
| **Deutsch-Jozsa** | Constant vs. balanced function classification | Exponential (Deterministic) | Parallel evaluation + Phase kickback | Artificial problem structure |
| **Bernstein-Vazirani** | Find hidden linear string $s$ in $f(x) = s \cdot x \pmod 2$ | Linear ($1$ query vs. $n$) | Hadamard decoding of parity phases | Oracle assumption is narrow |
| **Simon's** | Find hidden XOR period $s$ in 2-to-1 function | Exponential | GF(2) linear system sampling | Requires promised 2-to-1 symmetry |
| **Grover's** | Search unstructured database of size $N$ | Quadratic ($O(\sqrt{N})$) | Amplitude amplification via reflections | Oracle gate overhead & noise |
| **QFT** | Fourier transform over state amplitudes | Exponential | Controlled-phase rotation networks | Cannot directly measure amplitudes |
| **Shor's** | Prime factorization of composite $N$ | Superpolynomial | QFT + Modular Exponentiation | High logical qubit requirements |
| **Teleportation** | Quantum state transfer | N/A (Protocol) | Shared Bell pair + classical corrections | Requires physical channel for corrections |

---

## 3. Study Guide & Ordering

If you are new to quantum computing, we recommend studying the algorithms in the following sequential order to build intuition progressively:

```
  +--------------+     +----------------+     +---------------+
  | Entanglement | --> | Teleportation  | --> | Deutsch-Jozsa |
  +--------------+     +----------------+     +---------------+
                                                      |
                                                      v
  +--------------+     +----------------+     +---------------+
  |    Grover    | <-- |     Simon      | <-- |   B-Vazirani  |
  +--------------+     +----------------+     +---------------+
         |
         v
  +--------------+     +----------------+
  |     QFT      | --> |     Shor       |
  +--------------+     +----------------+
```

---

## 4. Core Quantum Mathematical Notation

### Single-Qubit Representation
A single qubit state $|\psi\rangle$ is represented as a unit vector in a 2-dimensional complex Hilbert space $\mathbb{C}^2$:
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle \quad \text{where } \alpha, \beta \in \mathbb{C} \text{ and } |\alpha|^2 + |\beta|^2 = 1$$
Upon measuring the state in the computational basis ($Z$-basis), we obtain:
* State $|0\rangle$ with probability $P(0) = |\alpha|^2$
* State $|1\rangle$ with probability $P(1) = |\beta|^2$

### Composite Multi-Qubit Registers
A system of $n$ qubits is represented in the tensor product space $\mathbb{C}^{2^n}$:
$$|\psi\rangle = \sum_{x \in \{0, 1\}^n} \alpha_x |x\rangle \quad \text{where } \sum_x |\alpha_x|^2 = 1$$
For example, a 2-qubit register is spanned by:
$$|00\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, \quad |01\rangle = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}, \quad |10\rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, \quad |11\rangle = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$$

---

## 5. Gate Mechanics Reference

| Gate Symbol | Operator Matrix | Operational Action | Description |
| :---: | :---: | :--- | :--- |
| **$X$** | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | $X\lvert0\rangle = \lvert1\rangle$, $X\lvert1\rangle = \lvert0\rangle$ | Pauli-X (Bit-flip) |
| **$Y$** | $\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ | $Y\lvert0\rangle = i\lvert1\rangle$, $Y\lvert1\rangle = -i\lvert0\rangle$ | Pauli-Y (Bit- & phase-flip) |
| **$Z$** | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | $Z\lvert0\rangle = \lvert0\rangle$, $Z\lvert1\rangle = -\lvert1\rangle$ | Pauli-Z (Phase-flip) |
| **$H$** | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | $H\lvert0\rangle = \lvert+\rangle$, $H\lvert1\rangle = \lvert-\rangle$ | Hadamard (Superposition creator) |
| **$S$** | $\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ | $S\lvert1\rangle = i\lvert1\rangle$ | Phase gate ($\pi/2$ rotation) |
| **$T$** | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$ | $T\lvert1\rangle = e^{i\pi/4}\lvert1\rangle$ | $\pi/4$ phase gate (Non-Clifford) |
| **$CX$** | 4x4 permutation | Flips target qubit $\iff$ control is $|1\rangle$ | Controlled-NOT (Entangling) |
| **$CZ$** | $\text{diag}(1, 1, 1, -1)$ | Flips phase of state $|11\rangle$ by $-1$ | Controlled-Phase |
| **$SWAP$**| 4x4 matrix | Swaps state values of two qubits | Register rearrangement |

---

## 6. Algorithmic Patterns

### Pattern A: Phase Kickback
Phase kickback is a fundamental subroutine used to load output information of a classical oracle function $f(x)$ directly into the phase of the input register. 

When the target qubit is prepared in the $|-\rangle$ state, querying the oracle $U_f |x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$ yields:
$$U_f |x\rangle |-\rangle = U_f |x\rangle \left( \frac{|0\rangle - |1\rangle}{\sqrt{2}} \right) = (-1)^{f(x)} |x\rangle |-\rangle$$
The state of the target qubit $|-\rangle$ remains unchanged, while the output factor $(-1)^{f(x)}$ is "kicked back" as a phase on the control register $|x\rangle$.

### Pattern B: Amplitude Amplification
Used in Grover's algorithm to rotate a starting uniform superposition $|\psi\rangle$ toward a target marked state $|w\rangle$. It alternates two geometric reflections:
1. **Oracle Reflection ($R_w$)**: Flips the sign of the marked state:
   $$R_w = I - 2|w\rangle\langle w|$$
2. **Diffusion Reflection ($R_s$)**: Reflects all states about the average amplitude:
   $$R_s = 2|s\rangle\langle s| - I \quad \text{where } |s\rangle = H^{\otimes n}|0\rangle$$

Each pair of reflections rotates the state vector closer to the target $|w\rangle$ in the 2D plane spanned by the target and the superposition of all non-target states.

### Pattern C: Fourier Sampling
The Quantum Fourier Transform maps states from the computational basis to the phase basis:
$$\text{QFT}|x\rangle = \frac{1}{\sqrt{N}} \sum_{y=0}^{N-1} e^{\frac{2\pi i x y}{N}} |y\rangle$$
By applying QFT (or its inverse), periodic structures hidden in the state amplitudes are mapped directly into measurable phase frequencies.

---

## 7. Glossary of Quantum Terms

* **Oracle (Black Box)**: A conceptual device that computes a function $f(x)$ on a quantum state without revealing its internal circuit structure.
* **Phase Kickback**: A technique where an oracle's output is written directly as a phase factor $(-1)^{f(x)}$ onto the input register by using an auxiliary qubit in the $|-\rangle$ state.
* **Promise Problem**: A problem where the input is guaranteed to come from a restricted subset of all possible inputs (e.g., in Deutsch-Jozsa, the oracle is *promised* to be either constant or balanced).
* **Amplitude Amplification**: The process of increasing the measurement probability of target states using constructive interference (as seen in Grover's search).
* **Modular Exponentiation**: The operation $f(x) = a^x \pmod N$. In Shor's algorithm, this represents the heavy quantum arithmetic step that creates the periodic state vector.
* **GF(2) Algebra**: Galois Field of order 2. Calculations are done modulo 2 (equivalent to XOR logic). Simon's algorithm uses this classical algebra to reconstruct the mask.
* **Continued Fractions**: A classical mathematical method used at the end of Shor's algorithm to convert the measured phase estimation fraction $y/2^t$ into the actual period integer $r$.
* **Unitary Operator**: A complex matrix $U$ satisfying $U^\dagger U = I$. It describes a reversible quantum computation.
* **Hermitian Operator**: A complex matrix $H$ satisfying $H^\dagger = H$. Represents a physical property (observable) that can be measured.
