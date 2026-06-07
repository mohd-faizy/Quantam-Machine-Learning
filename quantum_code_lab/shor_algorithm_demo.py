"""Small Shor-style period finding demo for factoring 15."""

from math import gcd


def period(a: int, n: int) -> int:
    value = 1
    for r in range(1, n * n):
        value = (value * a) % n
        if value == 1:
            return r
    raise ValueError("No period found.")


def shor_demo(n: int = 15, a: int = 2) -> tuple[int, int]:
    common = gcd(a, n)
    if common > 1:
        return common, n // common
    r = period(a, n)
    if r % 2 != 0:
        raise ValueError("Odd period; choose another a.")
    midpoint = pow(a, r // 2, n)
    return gcd(midpoint - 1, n), gcd(midpoint + 1, n)


if __name__ == "__main__":
    print(shor_demo())

