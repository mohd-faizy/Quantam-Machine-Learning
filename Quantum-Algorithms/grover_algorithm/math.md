# Grover Math

Let `N = 2^n` and one marked state be `|w>`. The uniform state is:

```text
|s> = 1/sqrt(N) sum_x |x>
```

Write:

```text
|s> = sin(theta)|w> + cos(theta)|r>
sin(theta) = 1/sqrt(N)
```

The oracle is:

```text
O_f|x> = (-1)^{f(x)}|x>
```

The diffusion operator is:

```text
D = 2|s><s| - I
```

One Grover iterate is `G = D O_f`. In the `span{|w>, |r>}` plane, each iterate rotates by `2 theta`. After `k` iterations:

```text
G^k |s> = sin((2k + 1)theta)|w> + cos((2k + 1)theta)|r>
```

Choose:

```text
k approx floor(pi/(4 theta)) approx floor(pi/4 sqrt(N))
```

so the marked measurement probability is near one.

