# Shor Algorithm

## Intuition

Shor's algorithm factors integers by reducing factoring to period finding. A quantum computer estimates the period of `a^x mod N` using phase estimation and the Quantum Fourier Transform (QFT); classical post-processing converts the period into factors.

## Step-by-Step

1. Choose an integer `a` coprime to `N`.
2. Build modular exponentiation `|x>|1> -> |x>|a^x mod N>`.
3. Use phase estimation and QFT to infer the period `r`.
4. If `r` is even and `a^(r/2) != -1 mod N`, compute `gcd(a^(r/2) +/- 1, N)`.
5. Retry with another `a` if post-processing fails.

## Circuit Diagram

```mermaid
graph LR
    A[Superposition over x] --> B[Modular exponentiation]
    B --> C[QFT inverse]
    C --> D[Measure phase]
    D --> E[Continued fractions]
    E --> F[Factors via gcd]
```

## Complexity

Polynomial in `log N` for the quantum period-finding core, compared with sub-exponential best-known classical factoring algorithms.

## Applications

Cryptanalysis of RSA-like systems, quantum complexity theory, phase estimation education, and post-quantum cryptography motivation.

## Limitations

Practical RSA-scale factoring requires fault-tolerant machines with many logical qubits. Toy demos use compiled modular arithmetic and do not imply near-term cryptographic threat.

