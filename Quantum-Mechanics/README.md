# Quantum Mechanics for Quantum Computing & QML

This module covers the physical foundations of quantum mechanics that translate directly to quantum computing and quantum machine learning (QML). Rather than standard physics derivations, we focus on the mathematical structures (Hilbert spaces, operators, statevectors, and density matrices) that form the building blocks of quantum software.

---

## Table of Contents
1. [Physical Intuition](#1-physical-intuition)
2. [The Core Postulates of Quantum Mechanics](#2-the-core-postulates-of-quantum-mechanics)
   - [Postulate 1: The State Space (Hilbert Space)](#postulate-1-the-state-space-hilbert-space)
   - [Postulate 2: Kinematics (Unitary Time Evolution)](#postulate-2-kinematics-unitary-time-evolution)
   - [Postulate 3: Measurement (Born's Rule & Collapse)](#postulate-3-measurement-borns-rule--collapse)
   - [Postulate 4: Composite Systems (Tensor Products)](#postulate-4-composite-systems-tensor-products)
3. [Key Quantum Phenomena in Computation](#3-key-quantum-phenomena-in-computation)
   - [Superposition & The Bloch Sphere](#superposition--the-bloch-sphere)
   - [Entanglement & Non-Locality](#entanglement--non-locality)
   - [Quantum Interference & Phase Mechanics](#quantum-interference--phase-mechanics)
4. [Mixed States, Density Matrices, and Noise](#4-mixed-states-density-matrices-and-noise)
   - [Pure vs. Mixed States](#pure-vs-mixed-states)
   - [The Density Operator ($\rho$)](#the-density-operator-rho)
   - [Partial Trace & Entanglement Measurement](#partial-trace--entanglement-measurement)
5. [The Quantum Mechanics of QML](#5-the-quantum-mechanics-of-qml)
   - [Parameterized Wavefunctions (VQCs)](#parameterized-wavefunctions-vqcs)
   - [Expectation Values as Loss Functions](#expectation-values-as-loss-functions)
   - [Quantum Feature Maps & Infinite-Dimensional Hilbert Space](#quantum-feature-maps--infinite-dimensional-hilbert-space)

---

## 1. Physical Intuition

In the classical world, physical objects possess definite, measurable properties at all times. A coin sitting on a table is either heads ($H$) or tails ($T$). A car is moving at exactly one velocity, and a ball is in exactly one position.

In the quantum world, physical systems at atomic scales (e.g., electrons, photons, superconducting circuits) behave differently:
* **No Definite State Before Measurement**: A quantum coin (e.g., a spinning electron's magnetic moment) does not choose to be spin-up or spin-down until it interacts with a measurement device. Prior to that, it exists in a **superposition** of both states.
* **Wave-Particle Duality**: Quantum entities exhibit both particle-like behavior (localized packets of energy) and wave-like behavior (interference patterns, distributed probabilities). 
* **State Space Expansion**: Adding classical bits increases a system's capacity *linearly* ($n$ bits can store one of $2^n$ configurations). Adding qubits increases the state space *exponentially* ($n$ qubits require $2^n$ complex amplitudes to describe, representing a vast, continuous state space).

---

## 2. The Core Postulates of Quantum Mechanics

The mathematical framework of quantum computing is built directly upon the four core postulates of quantum mechanics.

### Postulate 1: The State Space (Hilbert Space)
> **Postulate**: Any isolated physical system is associated with a complex vector space with an inner product (a **Hilbert Space** $\mathcal{H}$). The state of the system is completely described by a unit vector $|\psi\rangle$ in this space.

#### Dirac Notation (Bra-Ket Algebra)
We represent quantum states using Dirac notation:
* **Ket** $|\psi\rangle$: A column vector representing the quantum state.
  $$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$
* **Bra** $\langle\psi|$: The conjugate transpose (row vector) of $|\psi\rangle$. If $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$, then:
  $$\langle\psi| = \alpha^* \langle0| + \beta^* \langle1| = \begin{pmatrix} \alpha^* & \beta^* \end{pmatrix}$$
* **Inner Product** $\langle\phi|\psi\rangle$: A complex number representing the overlap or projection between two states.
  $$\langle\phi|\psi\rangle = \sum_i \phi_i^* \psi_i$$
  If $\langle\phi|\psi\rangle = 0$, the states are **orthogonal** (completely distinguishable).
* **Outer Product** $|\psi\rangle\langle\phi|$: An operator (matrix) mapping one state to another.
  $$|0\rangle\langle0| = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$
  The state vectors must be normalized, meaning their total probability sum is 1:
  $$\langle\psi|\psi\rangle = |\alpha|^2 + |\beta|^2 = 1$$

---

### Postulate 2: Kinematics (Unitary Time Evolution)
> **Postulate**: The time evolution of a closed quantum system is described by a **unitary operator** $U$. If the system is in state $|\psi_1\rangle$ at time $t_1$, its state $|\psi_2\rangle$ at time $t_2$ is:
> $$|\psi_2\rangle = U |\psi_1\rangle$$

A matrix $U$ is unitary if its conjugate transpose is its inverse:
$$U^\dagger U = U U^\dagger = I$$
* **Physical Significance**: Unitary evolution preserves the normalization of state vectors ($\langle\psi_2|\psi_2\rangle = \langle\psi_1|U^\dagger U|\psi_1\rangle = \langle\psi_1|\psi_1\rangle = 1$). This means probabilities always sum to 1, and quantum operations are **reversible** (we can reconstruct the input from the output by applying $U^\dagger$).
* **Circuit Representation**: Every quantum gate (e.g., Hadamard $H$, CNOT, Phase rotations) is represented by a unitary matrix acting on qubits.
* **Continuous Time**: The underlying physical time evolution is governed by the time-dependent **Schrödinger Equation**:
  $$i\hbar \frac{d}{dt}|\psi(t)\rangle = H(t)|\psi(t)\rangle$$
  where $H(t)$ is the **Hamiltonian** operator representing the total energy of the system. If $H$ is constant over time, $U = e^{-iHt/\hbar}$.

---

### Postulate 3: Measurement (Born's Rule & Collapse)
> **Postulate**: Quantum measurements are described by a collection of measurement operators $\{M_m\}$ acting on the state space. The index $m$ represents the possible measurement outcomes.
> The probability of obtaining outcome $m$ when measuring state $|\psi\rangle$ is:
> $$p(m) = \langle\psi|M_m^\dagger M_m|\psi\rangle$$
> If outcome $m$ occurs, the system's state immediately **collapses** to:
> $$|\psi'\rangle = \frac{M_m |\psi\rangle}{\sqrt{p(m)}}$$

#### Computational Basis Measurements
In a standard qubit measurement, we measure along the $Z$-basis ($|0\rangle$ and $|1\rangle$). The projection operators are:
$$M_0 = |0\rangle\langle0| = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \quad M_1 = |1\rangle\langle1| = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$$
For a state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$:
* **Probability of $|0\rangle$**: $p(0) = \langle\psi|M_0^\dagger M_0|\psi\rangle = \langle\psi|M_0|\psi\rangle = |\alpha|^2$
* **Probability of $|1\rangle$**: $p(1) = \langle\psi|M_1^\dagger M_1|\psi\rangle = \langle\psi|M_1|\psi\rangle = |\beta|^2$

#### Expectation Values
When we measure a physical observable represented by a Hermitian matrix $A$ (where $A^\dagger = A$), the average value (expectation value) obtained after many trials on identical states $|\psi\rangle$ is:
$$\langle A \rangle = \langle\psi|A|\psi\rangle$$
In QML, expectation values of Hamiltonian operators serve as the outputs of our quantum models.

---

### Postulate 4: Composite Systems (Tensor Products)
> **Postulate**: The state space of a composite physical system is the **tensor product** ($\otimes$) of the state spaces of the component systems. 

If we have system $A$ in state $|\psi_A\rangle$ and system $B$ in state $|\psi_B\rangle$, the joint system state is:
$$|\psi_{AB}\rangle = |\psi_A\rangle \otimes |\psi_B\rangle \quad (\text{often written as } |\psi_A\rangle|\psi_B\rangle \text{ or } |\psi_A \psi_B\rangle)$$

#### Example: Tensor Product of Two Qubits
If $|q_0\rangle = \alpha_0|0\rangle + \beta_0|1\rangle$ and $|q_1\rangle = \alpha_1|0\rangle + \beta_1|1\rangle$, the combined state is:
$$|q_0 q_1\rangle = \begin{pmatrix} \alpha_0 \\ \beta_0 \end{pmatrix} \otimes \begin{pmatrix} \alpha_1 \\ \beta_1 \end{pmatrix} = \begin{pmatrix} \alpha_0\alpha_1 \\ \alpha_0\beta_1 \\ \beta_0\alpha_1 \\ \beta_0\beta_1 \end{pmatrix} = \alpha_0\alpha_1|00\rangle + \alpha_0\beta_1|01\rangle + \beta_0\alpha_1|10\rangle + \beta_0\beta_1|11\rangle$$

The dimension of the composite Hilbert space scales as $2^n$ for $n$ qubits, leading to exponential computing capacity.

---

## 3. Key Quantum Phenomena in Computation

### Superposition & The Bloch Sphere
A classical bit is restricted to the binary extremes $0$ and $1$. A qubit, however, can exist in any linear combination:
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad \alpha, \beta \in \mathbb{C}, \quad |\alpha|^2 + |\beta|^2 = 1$$

Since absolute phase is unobservable, we can parameterize any single-qubit state using two real angles $\theta$ (polar angle) and $\phi$ (azimuthal angle):
$$|\psi\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi}\sin\left(\frac{\theta}{2}\right)|1\rangle, \quad 0 \le \theta \le \pi, \quad 0 \le \phi < 2\pi$$

This representation defines a point on the surface of a unit sphere called the **Bloch Sphere**:

* **North Pole ($\theta = 0$)**: State $|0\rangle$
* **South Pole ($\theta = \pi$)**: State $|1\rangle$
* **Equator ($\theta = \pi/2$)**: Equal superpositions, such as:
  * $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$ ($\phi = 0$)
  * $|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$ ($\phi = \pi$)
  * $|i+\rangle = \frac{|0\rangle + i|1\rangle}{\sqrt{2}}$ ($\phi = \pi/2$)

*See code lab example: [basic_quantum_circuits.py](../quantum_code_lab/basic_quantum_circuits.py)*

---

### Entanglement & Non-Locality
Entanglement occurs when a composite state **cannot** be factored into a tensor product of individual qubit states:
$$|\psi_{AB}\rangle \neq |\psi_A\rangle \otimes |\psi_B\rangle$$

#### Example: The Bell State $|\Phi^+\rangle$
$$|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$
If we measure the first qubit and get $|0\rangle$, the second qubit collapses to $|0\rangle$ instantaneously, regardless of physical distance. There is no combination of local states $|\psi_A\rangle = a_0|0\rangle + a_1|1\rangle$ and $|\psi_B\rangle = b_0|0\rangle + b_1|1\rangle$ that can generate $|\Phi^+\rangle$.

* **Why it matters**: Entanglement allows quantum algorithms to establish non-local correlations and coordinate calculations across qubits without exchanging classical signals.
* **QML Context**: Entangling layers in QML models build complex correlations across features, allowing the model to capture high-order interactions.

*See code lab example: [bell_states.py](../quantum_code_lab/bell_states.py)*

---

### Quantum Interference & Phase Mechanics
Quantum amplitudes $\alpha$ and $\beta$ are complex numbers, meaning they have a magnitude and a direction (phase). When we combine quantum paths, their amplitudes add:
* **Constructive Interference**: Amplitudes are in phase (same direction), increasing the probability of that outcome.
* **Destructive Interference**: Amplitudes are out of phase (opposite directions), canceling each other out and reducing the probability of that outcome.

$$\text{Probability of state } x = \left| \sum \text{path amplitudes} \right|^2$$

All quantum algorithms (e.g., Grover's Search, QFT, Shor's) work by using unitary gates to systematically align phases so that wrong answers interfere destructively (canceling out) and correct answers interfere constructively (maximizing measurement probability).

---

## 4. Mixed States, Density Matrices, and Noise

In practice, quantum computers are not perfectly isolated from their environment. Interaction with the surroundings causes noise, decoherence, and a loss of quantum information. To mathematically describe these noisy, open systems, we transition from state vectors to **Density Operators**.

### Pure vs. Mixed States
* **Pure State**: A state that can be described by a single state vector $|\psi\rangle$. We have 100% certainty about the quantum state vector.
* **Mixed State**: A statistical ensemble (probability distribution) of different pure states $\{p_i, |\psi_i\rangle\}$, where we are in state $|\psi_i\rangle$ with classical probability $p_i$ (such as 50% state $|0\rangle$ and 50% state $|1\rangle$).

---

### The Density Operator ($\rho$)
The density matrix $\rho$ of an ensemble is defined as:
$$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$

#### Properties of Density Matrices
1. **Hermitian**: $\rho^\dagger = \rho$
2. **Trace is 1**: $\text{Tr}(\rho) = 1$ (probabilities sum to 1)
3. **Positive Semidefinite**: $\langle\phi|\rho|\phi\rangle \ge 0$ for any vector $|\phi\rangle$
4. **Purity Test**: 
   * For pure states: $\text{Tr}(\rho^2) = 1$ (the density matrix is a projection operator, $\rho^2 = \rho$).
   * For mixed states: $\text{Tr}(\rho^2) < 1$.

#### Comparing Superposition and Classical Mixture
* **Superposition $|\psi\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$**:
  $$\rho_{\text{super}} = |\psi\rangle\langle\psi| = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & 1/2 \end{pmatrix}$$
  The off-diagonal terms ($1/2$) represent **quantum coherence** (phase relationships).
* **Classical Mixture (50% $|0\rangle$ and 50% $|1\rangle$)**:
  $$\rho_{\text{mix}} = \frac{1}{2}|0\rangle\langle0| + \frac{1}{2}|1\rangle\langle1| = \begin{pmatrix} 1/2 & 0 \\ 0 & 1/2 \end{pmatrix}$$
  The off-diagonal terms are $0$. Coherence has been lost due to interaction with the environment (decoherence).

---

### Partial Trace & Entanglement Measurement
If we have a bipartite system $AB$ described by a joint density matrix $\rho_{AB}$, the state of subsystem $A$ alone is obtained by "tracing out" system $B$:
$$\rho_A = \text{Tr}_B(\rho_{AB})$$

#### Tracing out Entanglement
If the joint state is the entangled Bell state $\rho_{AB} = |\Phi^+\rangle\langle\Phi^+|$:
$$\rho_{AB} = \frac{1}{2}\begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 1 \end{pmatrix}$$
Taking the partial trace over qubit $B$ yields a completely mixed state for qubit $A$:
$$\rho_A = \text{Tr}_B(\rho_{AB}) = \begin{pmatrix} 1/2 & 0 \\ 0 & 1/2 \end{pmatrix}$$
* **Insight**: An entangled joint system is pure, but its subsystems are mixed. The level of mixedness in $\rho_A$ (measured by **Von Neumann Entropy** $S(\rho_A) = -\text{Tr}(\rho_A \log \rho_A)$) is a direct measure of the entanglement between $A$ and $B$.

---

## 5. The Quantum Mechanics of QML

Quantum Machine Learning merges quantum mechanics with classical optimization. Here is how key QM concepts translate directly to machine learning components:

| Quantum Mechanics Concept | Mapping | Machine Learning Analogy |
| :--- | :---: | :--- |
| **Variational Wavefunction** $\lvert\psi(\theta)\rangle$ | $\rightarrow$ | Parameterized Model / Neural Network |
| **Hamiltonian Operator** $H$ | $\rightarrow$ | Loss Function / Optimization Objective |
| **Expectation Value** $\langle\psi(\theta)\rvert H \lvert\psi(\theta)\rangle$ | $\rightarrow$ | Model Output / Loss Function Value |
| **Hilbert Space Mapping** | $\rightarrow$ | Feature Map / Kernel Trick |

### Parameterized Wavefunctions (VQCs)
In Variational Quantum Algorithms (VQAs), we prepare a parameterized quantum state $|\psi(\theta)\rangle$ by passing a reference state (usually $|0\rangle$) through a circuit of gates controlled by tunable parameters $\theta$ (angles of rotation):
$$|\psi(\theta)\rangle = U(\theta)|0\rangle$$

The unitary matrix $U(\theta)$ acts as the model structure, while the parameters $\theta$ represent the trainable weights.

---

### Expectation Values as Loss Functions
In QML, we define our loss function $L(\theta)$ as the expectation value of an observable (Hamiltonian) $H$ representing our optimization task:
$$L(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle = \langle0|U^\dagger(\theta) H U(\theta)|0\rangle$$

#### Gradient Optimization (Parameter Shift Rule)
Unlike classical neural networks where gradients are computed via backpropagation, physical quantum hardware cannot track intermediate statevectors. Instead, we compute gradients using the **Parameter Shift Rule**:
$$\frac{\partial \langle H \rangle}{\partial \theta_i} = \frac{\langle H \rangle\left(\theta + \frac{\pi}{2}e_i\right) - \langle H \rangle\left(\theta - \frac{\pi}{2}e_i\right)}{2}$$
By shifting parameters forward and backward on the quantum processor, we obtain exact analytical gradients to feed into classical optimizers (Adam, COBYLA).

*See code lab example: [vqe_molecule_simulation.py](../quantum_code_lab/vqe_molecule_simulation.py)*

---

### Quantum Feature Maps & Infinite-Dimensional Hilbert Space
In classical Support Vector Machines (SVMs), the "kernel trick" maps low-dimensional data into a high-dimensional feature space where it becomes linearly separable.

Quantum Machine Learning does this naturally:
1. **Feature Map**: A unitary circuit encodes a classical data vector $x$ into a quantum state vector:
   $$|x\rangle = U(x)|0\rangle$$
2. **Quantum Kernel**: The similarity between two data points $x$ and $x'$ is represented by the inner product of their quantum states:
   $$K(x, x') = |\langle x|x'\rangle|^2 = |\langle0|U^\dagger(x) U(x')|0\rangle|^2$$

Because the dimension of the Hilbert space scales exponentially with the number of qubits ($2^n$), a quantum feature map can project data into an incredibly high-dimensional space. This allows quantum kernels to recognize complex patterns that would be computationally intractable to evaluate classically.

*See code lab example: [qml_hybrid_model.py](../quantum_code_lab/qml_hybrid_model.py)*

---

## 6. Quick Reference: Key Tables

### Classical vs. Quantum Systems

| Attribute | Classical Physics & Computing | Quantum Physics & Computing |
| :--- | :--- | :--- |
| **Basic Unit** | Bit ($0$ or $1$) | Qubit ($\alpha\lvert0\rangle + \beta\lvert1\rangle$) |
| **State Spaces** | Discrete states, capacity increases linearly | Hilbert space (complex vectors), capacity increases exponentially |
| **Time Evolution** | Deterministic/Boolean transformations | Continuous unitary matrix operations (reversible) |
| **Measurement** | Non-disruptive, yields exact physical properties | Disruptive (collapses state), probabilistic outcome |
| **Joint States** | Cartesian product (no entanglement possible) | Tensor product (allows maximally entangled states) |

### Standard Single-Qubit States

| State Name | Dirac Notation | State Vector | Bloch Coordinates $(\theta, \phi)$ |
| :--- | :---: | :---: | :---: |
| Ground / Zero | $\lvert0\rangle$ | $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$ | $(0, 0)$ |
| Excited / One | $\lvert1\rangle$ | $\begin{pmatrix} 0 \\ 1 \end{pmatrix}$ | $(\pi, 0)$ |
| Plus | $\lvert+\rangle$ | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$ | $(\pi/2, 0)$ |
| Minus | $\lvert-\rangle$ | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$ | $(\pi/2, \pi)$ |
| Right | $\lvert i+\rangle$ | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ i \end{pmatrix}$ | $(\pi/2, \pi/2)$ |
| Left | $\lvert i-\rangle$ | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -i \end{pmatrix}$ | $(\pi/2, 3\pi/2)$ |

### Physical Qubit Architectures

| Qubit Type | Physical Platform | State $\lvert0\rangle$ Definition | State $\lvert1\rangle$ Definition |
| :--- | :--- | :--- | :--- |
| **Superconducting** | Josephon junctions / LC circuits | Ground state energy levels | First excited state energy levels |
| **Trapped Ion** | Electromagnetic fields (e.g. Yb+) | Ground state hyperfine level | Excited hyperfine level (stable) |
| **Photonic** | Single photons | Horizontal polarization | Vertical polarization |
| **Neutral Atom** | Laser optical tweezers (e.g. Rubidium) | Ground state hyperfine level | High-energy Rydberg state |

---

## 7. Glossary of Quantum Terms

* **Hilbert Space ($\mathcal{H}$)**: A complete complex vector space equipped with an inner product. The mathematical space where quantum states reside.
* **Dirac Notation**: The standard bracket notation ($Bra$: $\langle\psi\rvert$, $Ket$: $\lvert\psi\rangle$) used to represent quantum state vectors and operators cleanly.
* **Unitary Operator ($U$)**: A matrix whose conjugate transpose is its inverse ($U^\dagger U = I$). Unitary operators preserve state normalization, meaning they are probability-conserving and reversible.
* **Hermitian Operator ($A$)**: A matrix that is equal to its own conjugate transpose ($A^\dagger = A$). In quantum mechanics, all physical observables (e.g., energy, spin, position) are represented by Hermitian operators because their eigenvalues are always real numbers.
* **Expectation Value ($\langle A \rangle$)**: The average measurement outcome for a physical observable $A$ when measured repeatedly over many copies of a quantum state $|\psi\rangle$, computed as $\langle\psi|A|\psi\rangle$.
* **Superposition**: The principle that a physical system can exist in a linear combination of multiple physical states simultaneously until a measurement is performed.
* **Quantum Coherence**: The presence of fixed phase relationships in a quantum superposition, represented by the off-diagonal elements of a density matrix.
* **Decoherence**: The loss of quantum coherence and transition of a pure state to a classical mixed state, caused by unwanted interaction/entanglement with the external environment.
* **Entanglement**: A purely quantum correlation where the state of a composite system cannot be written as a product of individual subsystem states. 
* **Density Matrix ($\rho$)**: A formulation of quantum states that can represent both pure states ($\text{Tr}(\rho^2) = 1$) and classical probability mixtures ($\text{Tr}(\rho^2) < 1$).
* **Partial Trace ($\text{Tr}_B(\rho_{AB})$)**: A mathematical operation that traces out subsystem $B$ from a joint system $AB$, leaving only the density matrix of subsystem $A$.
* **Parameter Shift Rule**: A method for evaluating analytical gradients of parameterized quantum circuits on physical hardware by shifting the target parameter by a constant amount ($\pm\pi/2$).
* **Quantum Feature Map**: A parameterized quantum circuit that projects classical data into quantum states in a high-dimensional Hilbert space.
* **Quantum Kernel**: The overlap or inner product similarity of two quantum feature mapped states, representing a distance metric in Hilbert space.
