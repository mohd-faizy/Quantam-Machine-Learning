# Deutsch-Jozsa Math

The oracle maps:

```text
U_f |x>|y> = |x>|y xor f(x)>
```

With `|-> = (|0> - |1>)/sqrt(2)`:

```text
U_f |x>|-> = (-1)^f(x)|x>|->
```

After the final Hadamards, the amplitude of `|0...0>` is:

```text
1/2^n sum_x (-1)^f(x)
```

This is `+1` or `-1` for constant functions and `0` for balanced functions.

