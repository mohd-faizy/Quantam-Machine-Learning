# Grover Theory

Grover's algorithm works in the two-dimensional subspace spanned by the marked-state vector `|w>` and the unmarked superposition `|r>`. The initial state is almost aligned with `|r>`. The oracle reflects the state across the unmarked axis by changing the phase of `|w>`. The diffusion operator reflects across the initial uniform state. Two reflections compose into a rotation, steadily increasing the amplitude of marked states.

The algorithm is powerful because probability amplitude can interfere. Wrong answers are not individually inspected; their amplitudes are collectively suppressed while the marked amplitude is amplified.

