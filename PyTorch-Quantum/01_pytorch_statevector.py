import math
import torch

"""
2-Qubit Statevector Simulator from First Principles using PyTorch.

This script demonstrates how to represent quantum states and apply quantum gates
using standard PyTorch complex tensors. It implements a simple Variational Quantum
Circuit (VQC) and uses PyTorch's native autograd engine to optimize the rotation
parameters to prepare a maximally entangled Bell state.

No external quantum libraries (such as Qiskit or PennyLane) are required, making this
100% compatible with modern Python 3.14+ environments.
"""

def get_initial_state():
    """
    Creates the initial state |00>.
    The state is represented as a tensor of shape (2, 2) where:
        state[i, j] is the amplitude of the basis state |i, j>
    with i being the state of qubit 0 and j being the state of qubit 1.
    """
    state = torch.zeros((2, 2), dtype=torch.complex64)
    state[0, 0] = 1.0 + 0.0j
    return state

def ry_matrix(theta):
    """
    Computes the 2x2 unitary matrix for the Y-rotation gate:
    RY(theta) = [[cos(theta/2), -sin(theta/2)],
                 [sin(theta/2),  cos(theta/2)]]
    
    Using torch.stack preserves the gradient flow through the parameter theta.
    """
    theta = theta.squeeze()
    cos_half = torch.cos(theta / 2.0)
    sin_half = torch.sin(theta / 2.0)
    
    row0 = torch.stack([cos_half, -sin_half])
    row1 = torch.stack([sin_half, cos_half])
    
    matrix = torch.stack([row0, row1])
    return matrix.to(torch.complex64)

def apply_gate_q0(state, gate):
    """
    Applies a 2x2 gate matrix to qubit 0.
    In a 2-qubit tensor state[i, j] representing |i, j>:
    Applying a gate G to qubit 0 corresponds to:
        new_state[i, j] = sum_k G[i, k] * state[k, j]
    This is equivalent to the matrix multiplication: G @ state
    """
    return torch.matmul(gate, state)

def apply_gate_q1(state, gate):
    """
    Applies a 2x2 gate matrix to qubit 1.
    In a 2-qubit tensor state[i, j] representing |i, j>:
    Applying a gate G to qubit 1 corresponds to:
        new_state[i, j] = sum_l G[j, l] * state[i, l]
    This is equivalent to the matrix multiplication: state @ G^T
    """
    return torch.matmul(state, gate.t())

def apply_cnot_01(state):
    """
    Applies CNOT where qubit 0 is the control and qubit 1 is the target.
    If qubit 0 is |1> (row 1), flip the state of qubit 1 (row 1 is flipped along dim 1).
    If qubit 0 is |0> (row 0), leave it unchanged.
    """
    new_state = torch.empty_like(state)
    # Control qubit 0 is |0> -> row 0 remains unchanged
    new_state[0, :] = state[0, :]
    # Control qubit 0 is |1> -> row 1 target is flipped
    new_state[1, :] = state[1, :].flip(0)
    return new_state

def calculate_fidelity(state, target_state):
    """
    Calculates the quantum fidelity F = |<state | target_state>|^2.
    """
    overlap = torch.sum(torch.conj(state) * target_state)
    return torch.abs(overlap) ** 2

def run_vqc(theta0, theta1):
    """
    Runs the Variational Quantum Circuit:
        |00> -> RY(theta0) on Q0 -> RY(theta1) on Q1 -> CNOT(0->1)
    """
    # Step 1: Initialize to |00>
    state = get_initial_state()
    
    # Step 2: Apply parameterized rotations
    ry0 = ry_matrix(theta0)
    ry1 = ry_matrix(theta1)
    state = apply_gate_q0(state, ry0)
    state = apply_gate_q1(state, ry1)
    
    # Step 3: Apply entangling CNOT gate
    state = apply_cnot_01(state)
    return state

def main():
    print("==================================================")
    print(" PyTorch Quantum Simulator - Variational Learning")
    print("==================================================")
    
    # Define target state: Bell State |Phi+> = 1/sqrt(2) * (|00> + |11>)
    target_state = torch.zeros((2, 2), dtype=torch.complex64)
    target_state[0, 0] = 1.0 / math.sqrt(2.0)
    target_state[1, 1] = 1.0 / math.sqrt(2.0)
    
    print("Target state matrix:\n", target_state)
    
    # Initialize trainable parameters with random values, requiring gradients
    torch.manual_seed(42)
    theta0 = torch.randn((), requires_grad=True)
    theta1 = torch.randn((), requires_grad=True)
    
    print(f"\nInitial Parameters:")
    print(f"  theta0: {theta0.item():.4f} rad")
    print(f"  theta1: {theta1.item():.4f} rad")
    
    # Initialize Adam optimizer
    optimizer = torch.optim.Adam([theta0, theta1], lr=0.1)
    
    epochs = 40
    print(f"\nOptimizing parameters to prepare Bell State (epochs={epochs})...")
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Run circuit simulation
        current_state = run_vqc(theta0, theta1)
        
        # Calculate fidelity and loss
        fidelity = calculate_fidelity(current_state, target_state)
        loss = 1.0 - fidelity
        
        # Backpropagate gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}: Loss = {loss.item():.6f} | Fidelity = {fidelity.item():.6f} | theta = [{theta0.item():.4f}, {theta1.item():.4f}]")
            
    print("\nOptimization Complete!")
    print(f"Final Parameters:")
    print(f"  theta0: {theta0.item():.4f} rad (Expected close to pi/2 ~ {math.pi/2:.4f})")
    print(f"  theta1: {theta1.item():.4f} rad (Expected close to 0.0)")
    
    # Final state evaluation
    final_state = run_vqc(theta0, theta1)
    print("\nFinal Statevector Matrix:\n", final_state)
    final_fidelity = calculate_fidelity(final_state, target_state)
    print(f"Final Fidelity to Bell State: {final_fidelity.item():.6%}")

if __name__ == "__main__":
    main()
