# Bernstein-Vazirani Algorithm

## Intuition

Bernstein-Vazirani finds a hidden bit string `s` from the oracle `f(x)=s dot x mod 2` using one quantum query.

## Step-by-Step

1. Prepare input register and an output `|->` state.
2. Put the input register in uniform superposition.
3. Query the oracle once.
4. Hadamard the input register.
5. Measure `s` directly.

## Circuit Diagram

```mermaid
graph LR
    A[Uniform superposition] --> B[Oracle f(x)=s dot x]
    B --> C[Hadamards]
    C --> D[Measure hidden string s]
```

## Complexity

One quantum query versus `n` classical queries.

## Applications

Hidden linear function learning, oracle model education, and phase kickback practice.

## Limitations

The speedup is query-based and assumes a clean oracle.

