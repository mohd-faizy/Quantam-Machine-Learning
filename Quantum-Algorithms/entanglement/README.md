# Entanglement

## Intuition

Entanglement is a nonclassical correlation where the joint state cannot be written as independent states of each subsystem. It is the resource behind teleportation, superdense coding, Bell tests, and many quantum speedups.

## Step-by-Step Bell State

1. Start with `|00>`.
2. Apply `H` to the first qubit.
3. Apply `CX` from first to second.
4. Measure both qubits to see correlated outcomes.

## Circuit Diagram

```mermaid
graph LR
    A[|00>] --> B[H on q0]
    B --> C[CX q0 to q1]
    C --> D[(|00> + |11>) / sqrt(2)]
```

## Complexity

Bell-state preparation uses one Hadamard and one entangling gate.

## Applications

Teleportation, QKD, quantum networking, error correction, and nonlocality experiments.

## Limitations

Entanglement is fragile under decoherence and cannot by itself enable faster-than-light communication.

