import torch
import torch.nn as nn
import torch.optim as optim

"""
Hybrid Classical-Quantum Classifier using PyTorch.

This script implements a hybrid classical-quantum model to solve the non-linear
XOR classification problem. It uses a classical Multi-Layer Perceptron (MLP)
to map inputs to quantum gate parameters (angles), passes them to a 2-qubit
first-principles simulator, and uses the expectation value of the final state
to make classification decisions.

Gradients flow back from the loss function, through the quantum simulator,
and into the classical MLP, demonstrating unified hybrid model optimization.
"""

def get_initial_state():
    """Creates the initial |00> state vector as a (2, 2) complex tensor."""
    state = torch.zeros((2, 2), dtype=torch.complex64)
    state[0, 0] = 1.0 + 0.0j
    return state

def ry_matrix(theta):
    """Computes the 2x2 RY(theta) matrix preserving gradient flow."""
    theta = theta.squeeze()
    cos_half = torch.cos(theta / 2.0)
    sin_half = torch.sin(theta / 2.0)
    
    row0 = torch.stack([cos_half, -sin_half])
    row1 = torch.stack([sin_half, cos_half])
    
    matrix = torch.stack([row0, row1])
    return matrix.to(torch.complex64)

def apply_gate_q0(state, gate):
    """Applies a 2x2 gate matrix to qubit 0."""
    return torch.matmul(gate, state)

def apply_gate_q1(state, gate):
    """Applies a 2x2 gate matrix to qubit 1."""
    return torch.matmul(state, gate.t())

def apply_cnot_01(state):
    """Applies CNOT with qubit 0 as control and qubit 1 as target."""
    new_state = torch.empty_like(state)
    new_state[0, :] = state[0, :]
    new_state[1, :] = state[1, :].flip(0)
    return new_state

class HybridQuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Classical feature mapping layer (MLP)
        # Maps 2 input coordinates to 2 rotation angles
        self.classical_mlp = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        # x is shape (batch_size, 2)
        batch_size = x.shape[0]
        predictions = []
        
        # Predict parameters for each sample in the batch
        angles = self.classical_mlp(x) # shape (batch_size, 2)
        
        for i in range(batch_size):
            theta0 = angles[i, 0]
            theta1 = angles[i, 1]
            
            # Run the quantum circuit simulation
            state = get_initial_state()
            ry0 = ry_matrix(theta0)
            ry1 = ry_matrix(theta1)
            
            state = apply_gate_q0(state, ry0)
            state = apply_gate_q1(state, ry1)
            state = apply_cnot_01(state)
            
            # Calculate expectation value of Z observable on qubit 1 (Z1)
            # Z1 = I x Z. Expectation is P(|00>) + P(|10>) - P(|01>) - P(|11>)
            probs = torch.abs(state) ** 2
            exp_z1 = probs[0, 0] + probs[1, 0] - probs[0, 1] - probs[1, 1]
            predictions.append(exp_z1)
            
        return torch.stack(predictions)

def main():
    print("==================================================")
    print(" Hybrid Classical-Quantum XOR Classifier")
    print("==================================================")
    
    # Dataset: XOR Problem
    # Inputs: (x0, x1)
    # Labels: -1.0 for same inputs, +1.0 for different inputs
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([-1.0, 1.0, 1.0, -1.0], dtype=torch.float32)
    
    model = HybridQuantumClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()
    
    epochs = 100
    print(f"Training hybrid model for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)
        
        # Backward pass & update
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: Loss = {loss.item():.6f}")
            
    print("\nTraining Complete!")
    
    # Evaluate model predictions
    with torch.no_grad():
        final_preds = model(X)
        print("\nFinal Model Predictions:")
        for idx in range(4):
            input_val = X[idx].tolist()
            pred_val = final_preds[idx].item()
            target_val = y[idx].item()
            print(f"  Input: {input_val} | Prediction: {pred_val:+.4f} | Target: {target_val:+.1f}")
            
if __name__ == "__main__":
    main()
