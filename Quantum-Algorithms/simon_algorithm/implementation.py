"""Classical GF(2) post-processing sketch for Simon samples."""


def dot_mod2(a: str, b: str) -> int:
    return sum(int(x) & int(y) for x, y in zip(a, b)) % 2


def candidates(samples: list[str]) -> list[str]:
    n = len(samples[0])
    out = []
    for value in range(1, 2**n):
        s = format(value, f"0{n}b")
        if all(dot_mod2(s, z) == 0 for z in samples):
            out.append(s)
    return out


if __name__ == "__main__":
    print(candidates(["001", "010"]))

