# Quantum Teleportation

## Intuition

Quantum teleportation transfers an unknown qubit state using shared entanglement plus two classical bits. The original state is not copied; measurement destroys local coherence while correction reconstructs the state remotely.

## Step-by-Step

1. Prepare unknown state `|psi>`.
2. Create a Bell pair between sender and receiver.
3. Entangle `|psi>` with sender's Bell qubit.
4. Measure sender qubits.
5. Send two classical bits.
6. Receiver applies conditional `X` and `Z` corrections.

## Circuit Diagram

```mermaid
graph LR
    A[Unknown state] --> B[Bell-basis measurement]
    C[Shared Bell pair] --> B
    B --> D[Classical bits]
    D --> E[Conditional X/Z correction]
    E --> F[Recovered state]
```

## Complexity

Requires one entangled pair, two classical bits, and local gates.

## Applications

Quantum networking, repeaters, distributed quantum computing, and fault-tolerant gate teleportation.

## Limitations

Requires entanglement distribution and classical communication; it does not transmit information faster than light.

