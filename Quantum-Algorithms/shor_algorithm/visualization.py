"""Print the Shor period-finding pipeline as a Mermaid graph."""


def main() -> None:
    print(
        "graph LR\n"
        "A[Choose a coprime base] --> B[Quantum period finding]\n"
        "B --> C[Continued fractions]\n"
        "C --> D[gcd post-processing]\n"
        "D --> E[Candidate factors]\n"
    )


if __name__ == "__main__":
    main()

