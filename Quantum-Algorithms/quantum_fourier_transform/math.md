# QFT Math

For `N=2^n`:

```text
QFT |x> = 1/sqrt(N) sum_y exp(2 pi i x y / N) |y>
```

The binary expansion form decomposes into single-qubit phase states:

```text
QFT |x1...xn> = 1/sqrt(2^n) tensor_k (|0> + exp(2 pi i 0.x_k...x_n)|1>)
```

This factorization enables an `O(n^2)` circuit of Hadamards and controlled phase rotations.

