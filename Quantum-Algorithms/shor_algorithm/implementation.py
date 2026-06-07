"""Educational Shor demo for N=15 using classical period post-processing."""

from __future__ import annotations

from math import gcd


def find_period_classically(a: int, n: int) -> int:
    value = 1
    for r in range(1, n * n):
        value = (value * a) % n
        if value == 1:
            return r
    raise ValueError("period not found")


def factor_demo(n: int = 15, a: int = 2) -> tuple[int, int]:
    if gcd(a, n) != 1:
        return gcd(a, n), n // gcd(a, n)
    r = find_period_classically(a, n)
    if r % 2:
        raise ValueError("period is odd; choose another base")
    x = pow(a, r // 2, n)
    return gcd(x - 1, n), gcd(x + 1, n)


def main() -> None:
    print(f"Factors of 15 from period finding: {factor_demo()}")


if __name__ == "__main__":
    main()

