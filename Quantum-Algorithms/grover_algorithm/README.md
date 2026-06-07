# Grover Algorithm

## Intuition

Grover's algorithm searches an unstructured space of `N` candidates using amplitude amplification. Instead of checking candidates one by one, it rotates the quantum state toward the marked answer so measurement becomes likely after about `sqrt(N)` oracle calls.

## Step-by-Step

1. Prepare a uniform superposition with Hadamard gates.
2. Apply an oracle that flips the phase of marked states.
3. Apply the diffusion operator to reflect amplitudes about their mean.
4. Repeat oracle plus diffusion about `pi/4 * sqrt(N/M)` times for `M` marked states.
5. Measure and classically verify the result.

## Circuit Diagram

```mermaid
graph LR
    A[|0...0>] --> B[Hadamards]
    B --> C[Oracle phase flip]
    C --> D[Diffusion]
    D --> E{Repeat}
    E --> C
    E --> F[Measure]
```

## Complexity

| Resource | Classical | Grover |
|---|---:|---:|
| Oracle queries | O(N) | O(sqrt(N)) |
| Space | O(log N) index bits | O(log N) qubits plus oracle workspace |

## Applications

Database search, SAT-style oracle search, cryptographic key search, combinatorial optimization subroutines, and amplitude amplification.

## Limitations

The oracle must be efficient, the quadratic speedup is not exponential, and noise limits useful iteration depth on NISQ hardware.

