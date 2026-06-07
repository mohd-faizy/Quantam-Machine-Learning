# Quantum Algorithms Knowledge System

This directory replaces PDF/wiki dumps with modular, reviewable algorithm notes. Each algorithm folder follows the same structure:

- `README.md`: intuition, circuit map, complexity, applications, limitations.
- `theory.md`: conceptual explanation and reason the algorithm works.
- `math.md`: linear algebra, Dirac notation, and derivation.
- `implementation.py`: runnable Python implementation with graceful dependency notes.
- `visualization.py`: circuit or state visualization entry point.
- `circuits/`: room for exported diagrams, QASM, and circuit snapshots.

## Suggested Study Order

1. Entanglement and teleportation.
2. Deutsch-Jozsa and Bernstein-Vazirani.
3. Simon's algorithm.
4. Grover search.
5. Quantum Fourier Transform.
6. Shor's algorithm.
7. Variational algorithms in `../qml` and `../quantum_code_lab`.

