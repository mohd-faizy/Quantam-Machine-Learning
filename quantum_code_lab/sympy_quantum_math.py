"""Symbolic quantum math examples with SymPy."""


def grover_rotation_formula() -> str:
    try:
        import sympy as sp
    except ImportError as exc:
        raise SystemExit("Install the unified stack: pip install -r requirements.txt") from exc

    n, k = sp.symbols("N k", positive=True)
    theta = sp.asin(1 / sp.sqrt(n))
    success_probability = sp.sin((2 * k + 1) * theta) ** 2
    return str(sp.simplify(success_probability))


def bell_state_density_matrix() -> str:
    try:
        import sympy as sp
    except ImportError as exc:
        raise SystemExit("Install the unified stack: pip install -r requirements.txt") from exc

    ket = sp.Matrix([1, 0, 0, 1]) / sp.sqrt(2)
    rho = ket * ket.T
    return str(rho)


if __name__ == "__main__":
    print("Grover success probability:")
    print(grover_rotation_formula())
    print("\nBell density matrix:")
    print(bell_state_density_matrix())

