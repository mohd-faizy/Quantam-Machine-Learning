# Shor Math

For `gcd(a, N) = 1`, define:

```text
f(x) = a^x mod N
```

The period `r` satisfies `f(x + r) = f(x)`. After modular exponentiation:

```text
1/sqrt(Q) sum_x |x>|a^x mod N>
```

Measuring the second register leaves an arithmetic progression in the first. Applying inverse QFT produces likely outcomes near:

```text
y / Q approx s / r
```

Continued fractions recover `r`. If `r` is even:

```text
a^r - 1 = (a^(r/2) - 1)(a^(r/2) + 1) = 0 mod N
```

Nontrivial factors are:

```text
gcd(a^(r/2) - 1, N), gcd(a^(r/2) + 1, N)
```

