"""
Shor's Algorithm — Complete Implementation
===========================================

Factors composite integers by reducing the problem to quantum period finding.
A quantum computer estimates the period r of the modular exponentiation
function f(x) = aˣ mod N using phase estimation and the Quantum Fourier
Transform.  Classical post-processing (continued fractions) then converts
the period into factors of N.

This implementation provides:
    1. A full quantum order-finding circuit for N=15 (compiled oracle)
    2. Classical period-finding for comparison and larger numbers
    3. The complete Shor pipeline: random base selection → quantum period
       finding → continued fractions → GCD factor extraction
    4. Verification against known factorizations

Features
--------
    • Quantum order-finding circuit for N=15 with compiled modular exponentiation
    • Inverse QFT for phase readout
    • Continued fractions algorithm for period extraction
    • Full classical Shor pipeline for arbitrary N
    • Multiple base attempts with retry logic

Usage
-----
    python implementation.py
"""

from __future__ import annotations

import math
import random
from fractions import Fraction


def _check_qiskit():
    """Ensure Qiskit is installed and return required modules."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        return QuantumCircuit, Statevector
    except ImportError as exc:
        raise SystemExit(
            "Qiskit is required.  Install it with:\n"
            "  pip install 'qiskit>=1.0' qiskit-aer"
        ) from exc


# ---------------------------------------------------------------------------
# Classical helpers
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Simple primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def is_prime_power(n: int) -> tuple[bool, int, int]:
    """Check if n = p^k for some prime p and k ≥ 2.

    Returns (True, p, k) or (False, 0, 0).
    """
    for k in range(2, int(math.log2(n)) + 1):
        p = round(n ** (1 / k))
        for candidate in [p - 1, p, p + 1]:
            if candidate > 1 and candidate ** k == n:
                if is_prime(candidate):
                    return True, candidate, k
    return False, 0, 0


def find_period_classically(a: int, N: int) -> int:
    """Find the period r such that a^r ≡ 1 (mod N) by brute force.

    Parameters
    ----------
    a : int
        Base (must be coprime to N).
    N : int
        Modulus.

    Returns
    -------
    int
        The order r of a modulo N.
    """
    value = 1
    for r in range(1, N * N):
        value = (value * a) % N
        if value == 1:
            return r
    raise ValueError(f"Period not found for a={a}, N={N}")


def extract_factors(a: int, r: int, N: int) -> tuple[int, int] | None:
    """Extract factors of N from the period r of a^x mod N.

    Returns
    -------
    tuple[int, int] or None
        Non-trivial factors (p, q), or None if this period doesn't yield factors.
    """
    if r % 2 != 0:
        return None  # Need even period

    x = pow(a, r // 2, N)
    if x == N - 1:
        return None  # a^{r/2} ≡ −1 (mod N) — trivial

    p = math.gcd(x - 1, N)
    q = math.gcd(x + 1, N)

    if p == 1 or p == N:
        p = None
    if q == 1 or q == N:
        q = None

    if p and q and p * q == N:
        return (min(p, q), max(p, q))
    elif p and p != N:
        return (p, N // p)
    elif q and q != N:
        return (q, N // q)
    return None


# ---------------------------------------------------------------------------
# Quantum circuits for N=15
# ---------------------------------------------------------------------------

def build_mod_exp_gate(a: int, power: int, N: int = 15) -> "QuantumCircuit":
    """Build a compiled modular exponentiation gate for N=15.

    Implements the unitary |y⟩ → |a^{power} · y mod 15⟩ on 4 qubits.

    For N=15 with specific bases, these are compiled into simple SWAP networks
    rather than full arithmetic circuits.
    """
    QuantumCircuit, _ = _check_qiskit()

    qc = QuantumCircuit(4, name=f"{a}^{power} mod {N}")
    a_pow = pow(a, power, N)

    if a_pow == 1:
        return qc  # Identity — no gates needed

    # Compiled permutations for specific values of a^power mod 15
    # These implement |y⟩ → |a^power · y mod 15⟩ for y in {1, 2, 4, 7, 8, 11, 13, 14}
    if a_pow == 2:
        qc.swap(0, 1)
        qc.swap(1, 2)
        qc.swap(2, 3)
    elif a_pow == 4:
        qc.swap(0, 2)
        qc.swap(1, 3)
    elif a_pow == 7:
        qc.swap(2, 3)
        qc.swap(1, 2)
        qc.swap(0, 1)
        qc.x(range(4))
    elif a_pow == 8:
        qc.swap(2, 3)
        qc.swap(1, 2)
        qc.swap(0, 1)
    elif a_pow == 11:
        qc.swap(0, 1)
        qc.swap(1, 2)
        qc.swap(2, 3)
        qc.x(range(4))
    elif a_pow == 13:
        qc.swap(0, 2)
        qc.swap(1, 3)
        qc.x(range(4))
    elif a_pow == 14:
        qc.x(range(4))  # equivalent to ×14 mod 15 since 14 ≡ −1
    else:
        raise ValueError(f"Unsupported a^power mod 15 = {a_pow}")

    return qc


def build_inverse_qft(n: int) -> "QuantumCircuit":
    """Build the inverse QFT circuit for phase readout."""
    QuantumCircuit, _ = _check_qiskit()

    qc = QuantumCircuit(n, name=f"IQFT({n})")

    # Reverse qubit order first
    for i in range(n // 2):
        qc.swap(i, n - i - 1)

    # Inverse QFT gates (reversed order, negative angles)
    for target in range(n - 1, -1, -1):
        for control in range(n - 1, target, -1):
            k = control - target
            qc.cp(-math.pi / (2 ** k), control, target)
        qc.h(target)

    return qc


def build_shor_circuit(a: int = 7, N: int = 15, n_count: int = 8) -> "QuantumCircuit":
    """Build the full quantum order-finding circuit for Shor's algorithm.

    Parameters
    ----------
    a : int
        Base for modular exponentiation (must be coprime to N).
    N : int
        Number to factor.  Currently only N=15 is supported with compiled gates.
    n_count : int
        Number of counting qubits (determines phase precision).

    Returns
    -------
    QuantumCircuit
        Circuit with n_count counting qubits + 4 work qubits.

    Circuit structure
    -----------------
    1. Initialise work register to |1⟩ (since a⁰ = 1).
    2. Hadamard all counting qubits → uniform superposition over exponents.
    3. Controlled modular exponentiation: for each counting qubit k,
       apply controlled-U^{2^k} where U|y⟩ = |a·y mod N⟩.
    4. Inverse QFT on counting register → phase → period information.
    5. Measure counting register.
    """
    if N != 15:
        raise NotImplementedError(
            f"Compiled quantum circuit only supports N=15 (got N={N}).  "
            "Use find_period_classically() for other values."
        )
    if math.gcd(a, N) != 1:
        raise ValueError(f"a={a} is not coprime to N={N}.")

    QuantumCircuit, _ = _check_qiskit()

    n_work = 4  # qubits for the work register (enough for mod 15)
    total = n_count + n_work

    qc = QuantumCircuit(total, n_count, name=f"Shor(a={a}, N={N})")

    # Step 1 — Initialise work register to |1⟩ = |0001⟩
    qc.x(n_count)  # work qubit 0 (= qubit index n_count)

    # Step 2 — Hadamard all counting qubits
    qc.h(range(n_count))
    qc.barrier(label="superposition")

    # Step 3 — Controlled modular exponentiation
    for k in range(n_count):
        power = 2 ** k
        mod_exp = build_mod_exp_gate(a, power, N)
        controlled_mod_exp = mod_exp.to_gate().control(1)
        # Control qubit is k, target qubits are n_count to n_count+3
        qc.append(controlled_mod_exp,
                   [k] + list(range(n_count, n_count + n_work)))
    qc.barrier(label="mod exp")

    # Step 4 — Inverse QFT on counting register
    iqft = build_inverse_qft(n_count)
    qc.compose(iqft, qubits=range(n_count), inplace=True)
    qc.barrier(label="IQFT")

    # Step 5 — Measure counting register
    qc.measure(range(n_count), range(n_count))

    return qc


# ---------------------------------------------------------------------------
# Period extraction from measurement
# ---------------------------------------------------------------------------

def extract_period_from_phase(measured_value: int, n_count: int, N: int) -> int | None:
    """Extract the period from a phase measurement using continued fractions.

    Parameters
    ----------
    measured_value : int
        The integer measured from the counting register.
    n_count : int
        Number of counting qubits.
    N : int
        The number being factored.

    Returns
    -------
    int or None
        Candidate period, or None if extraction fails.
    """
    if measured_value == 0:
        return None

    Q = 2 ** n_count
    phase = measured_value / Q

    # Use continued fractions to find s/r ≈ phase with r < N
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator

    if r < 1 or r >= N:
        return None

    return r


# ---------------------------------------------------------------------------
# Full Shor pipeline
# ---------------------------------------------------------------------------

def run_shor_quantum(N: int = 15, a: int = 7, n_count: int = 8,
                      shots: int = 2048) -> dict:
    """Run the quantum part of Shor's algorithm for N=15.

    Returns
    -------
    dict with keys: 'circuit', 'counts', 'candidate_periods', 'factors'
    """
    _, Statevector = _check_qiskit()
    QuantumCircuit, _ = _check_qiskit()

    # Build unitary part (no measurement) for statevector simulation
    n_work = 4
    total = n_count + n_work

    qc_sv = QuantumCircuit(total, name=f"Shor-sv(a={a}, N={N})")
    qc_sv.x(n_count)
    qc_sv.h(range(n_count))

    for k in range(n_count):
        power = 2 ** k
        mod_exp = build_mod_exp_gate(a, power, N)
        controlled_mod_exp = mod_exp.to_gate().control(1)
        qc_sv.append(controlled_mod_exp,
                      [k] + list(range(n_count, n_count + n_work)))

    iqft = build_inverse_qft(n_count)
    qc_sv.compose(iqft, qubits=range(n_count), inplace=True)

    sv = Statevector.from_instruction(qc_sv)
    raw_counts = sv.sample_counts(shots)

    # Extract counting register bits
    count_results: dict[str, int] = {}
    for bitstring, count in raw_counts.items():
        # Counting register = last n_count bits (Qiskit ordering)
        count_bits = bitstring[n_work:]
        count_results[count_bits] = count_results.get(count_bits, 0) + count

    # Extract candidate periods
    candidate_periods: dict[int, int] = {}
    for bitstring, count in count_results.items():
        measured_val = int(bitstring, 2)
        r = extract_period_from_phase(measured_val, n_count, N)
        if r is not None and r > 0:
            candidate_periods[r] = candidate_periods.get(r, 0) + count

    # Try to extract factors from each candidate period
    factors = None
    for r in sorted(candidate_periods, key=candidate_periods.get, reverse=True):
        result = extract_factors(a, r, N)
        if result is not None:
            factors = result
            break

    return {
        "circuit": build_shor_circuit(a, N, n_count),
        "counts": count_results,
        "candidate_periods": candidate_periods,
        "factors": factors,
    }


def run_shor_classical(N: int, max_attempts: int = 20) -> tuple[int, int] | None:
    """Run the full Shor pipeline using classical period finding.

    Works for arbitrary composite N (not just 15).
    """
    if N < 4 or is_prime(N) or N % 2 == 0:
        if N % 2 == 0:
            return (2, N // 2)
        return None

    pp, p, k = is_prime_power(N)
    if pp:
        return (p, N // p)

    for _ in range(max_attempts):
        a = random.randint(2, N - 1)
        g = math.gcd(a, N)
        if g > 1:
            return (g, N // g)

        r = find_period_classically(a, N)
        result = extract_factors(a, r, N)
        if result is not None:
            return result

    return None


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Shor's algorithm demonstration."""
    print("=" * 70)
    print("  SHOR'S ALGORITHM — INTEGER FACTORIZATION")
    print("=" * 70)

    N = 15
    a = 7
    n_count = 8
    SHOTS = 4096

    # --- 1. Classical period finding for reference ---
    print(f"\n▸ CLASSICAL PERIOD FINDING (N={N})\n")
    for base in [2, 4, 7, 8, 11, 13]:
        if math.gcd(base, N) == 1:
            r = find_period_classically(base, N)
            factors = extract_factors(base, r, N)
            status = f"→ factors {factors}" if factors else "→ no factors (period odd or trivial)"
            print(f"  a={base:2d}:  r={r}  {status}")

    # --- 2. Quantum circuit ---
    print(f"\n▸ QUANTUM ORDER-FINDING CIRCUIT (a={a}, N={N}, {n_count} counting qubits)\n")
    qc = build_shor_circuit(a, N, n_count)
    print(f"  Circuit: {qc.num_qubits} qubits, {qc.size()} gates, depth={qc.depth()}")
    print()

    # --- 3. Quantum simulation ---
    print(f"▸ QUANTUM SIMULATION ({SHOTS:,} shots)\n")
    result = run_shor_quantum(N, a, n_count, shots=SHOTS)

    print(f"  Measurement results (counting register):")
    sorted_counts = sorted(result["counts"].items(), key=lambda x: -x[1])
    for bitstring, count in sorted_counts[:12]:
        decimal = int(bitstring, 2)
        phase = decimal / (2 ** n_count)
        bar = "█" * int(30 * count / SHOTS)
        print(f"    |{bitstring}⟩ = {decimal:3d}  "
              f"(phase ≈ {phase:.4f})  {count:5d}  {bar}")

    print(f"\n  Candidate periods (from continued fractions):")
    for r, count in sorted(result["candidate_periods"].items(),
                            key=lambda x: -x[1]):
        factors = extract_factors(a, r, N)
        f_str = f"→ factors {factors}" if factors else "→ no factors"
        print(f"    r={r}:  {count} votes  {f_str}")

    if result["factors"]:
        p, q = result["factors"]
        print(f"\n  ★ RESULT: {N} = {p} × {q}")
        if p * q == N:
            print(f"  ✓ Verified: {p} × {q} = {N}")
        else:
            print(f"  ✗ Verification failed!")
    else:
        print(f"\n  ✗ No factors found — try a different base.")

    # --- 4. Classical Shor on larger numbers ---
    print(f"\n{'─' * 70}")
    print(f"  CLASSICAL SHOR PIPELINE (larger numbers)")
    print(f"{'─' * 70}\n")

    test_numbers = [15, 21, 33, 35, 51, 77, 91, 143]
    for n in test_numbers:
        factors = run_shor_classical(n)
        if factors:
            p, q = factors
            verify = "✓" if p * q == n else "✗"
            print(f"  {n:4d} = {p} × {q}  {verify}")
        else:
            print(f"  {n:4d} = (factoring failed)")

    print("\n" + "=" * 70)
    print("  ★ Shor's algorithm: polynomial-time factoring on a quantum computer.")
    print("  ★ Threatens RSA cryptography when fault-tolerant hardware is available.")
    print("=" * 70)


if __name__ == "__main__":
    main()
