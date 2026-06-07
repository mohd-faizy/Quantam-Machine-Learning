# Deutsch-Jozsa Algorithm

## Intuition

Deutsch-Jozsa distinguishes a constant Boolean function from a balanced one with one quantum oracle query.

## Step-by-Step

1. Prepare input qubits in `|0...0>` and an output qubit in `|1>`.
2. Apply Hadamards to create phase kickback.
3. Query the oracle once.
4. Apply Hadamards to input qubits.
5. Measure: all zeros means constant; any one bit means balanced.

## Circuit Diagram

```mermaid
graph LR
    A[Prepare |0...0>|1>] --> B[Hadamards]
    B --> C[Oracle U_f]
    C --> D[Hadamards on input]
    D --> E[Measure]
```

## Complexity

One quantum query versus deterministic classical worst-case `2^(n-1)+1` queries.

## Applications

Oracle separation, phase kickback teaching, and early demonstration of quantum query advantage.

## Limitations

The promise problem is artificial; practical value is mainly conceptual.

