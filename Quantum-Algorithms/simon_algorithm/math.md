# Simon Math

After measuring the function register:

```text
(|x> + |x xor s>) / sqrt(2)
```

Applying `H^n` gives amplitudes proportional to:

```text
(-1)^(x dot z) [1 + (-1)^(s dot z)]
```

The amplitude is zero unless:

```text
s dot z = 0 mod 2
```

Collect `n-1` independent equations to solve for nonzero `s`.

