# Simon Algorithm

## Intuition

Simon finds a hidden XOR mask `s` where `f(x)=f(y)` exactly when `y=x xor s`. It was an early exponential oracle separation and inspired Shor's period-finding work.

## Step-by-Step

1. Prepare two `n`-qubit registers.
2. Put the first register into uniform superposition.
3. Query the two-to-one oracle.
4. Measure or ignore the second register.
5. Hadamard the first register and measure equations `z dot s = 0`.
6. Solve the linear system over GF(2).

## Circuit Diagram

```mermaid
graph LR
    A[Superposition over x] --> B[Oracle f(x)]
    B --> C[Measure function register]
    C --> D[Hadamards]
    D --> E[Linear equations over GF(2)]
```

## Complexity

Polynomial quantum queries versus exponential randomized classical queries in the oracle model.

## Applications

Hidden subgroup problems, period-finding intuition, and quantum algorithm theory.

## Limitations

Requires a promise oracle and classical GF(2) post-processing.

