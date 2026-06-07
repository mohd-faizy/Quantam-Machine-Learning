import sys
import math
import torch
import torch.nn as nn

"""
MIT HAN Lab torchquantum Legacy Syntax Showcase.

This script demonstrates the syntax and architecture of the official 'torchquantum'
library. It implements a Quantum Neural Network (QNN) that prepares a Bell state
using 'tq.QuantumModule', 'tq.QuantumDevice', and parameterized gates.

Since torchquantum relies on pre-1.0 legacy versions of Qiskit that do not support
modern Python 3.14+ or Qiskit 2.x, this script gracefully checks for the library and:
1. Executes the official training loop if 'torchquantum' is installed.
2. Displays a detailed diagnostics warning and installation guide if it is missing,
   preventing abrupt import crashes.
"""

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    TORCHQUANTUM_AVAILABLE = True
except ImportError:
    TORCHQUANTUM_AVAILABLE = False


class TorchQuantumQNN(nn.Module):
    """
    A Quantum Neural Network Module implemented using official torchquantum syntax.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        
        # In torchquantum, states are stored inside a QuantumDevice object
        if TORCHQUANTUM_AVAILABLE:
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        else:
            self.q_device = None
            
        # Define parameterized angles as PyTorch nn.Parameters
        self.theta0 = nn.Parameter(torch.tensor(0.3367))  # Initial angle Q0
        self.theta1 = nn.Parameter(torch.tensor(0.1288))  # Initial angle Q1

    def forward(self):
        """
        Executes the quantum circuit:
            |00> -> RY(theta0) on Q0 -> RY(theta1) on Q1 -> CNOT(0 -> 1)
        """
        if not TORCHQUANTUM_AVAILABLE:
            raise RuntimeError("torchquantum is not installed or importable in this environment.")
            
        # 1. Reset the statevector of the quantum device to the |00> state
        self.q_device.reset_states(bsz=1)
        
        # 2. Apply parameterized Y-rotation on qubit 0 and qubit 1
        tqf.ry(self.q_device, wires=0, params=self.theta0.unsqueeze(0))
        tqf.ry(self.q_device, wires=1, params=self.theta1.unsqueeze(0))
        
        # 3. Apply CNOT gate (wires parameter accepts [control, target])
        tqf.cnot(self.q_device, wires=[0, 1])
        
        # 4. Return the statevector of the device (shape: [bsz, 2^n_wires])
        return self.q_device.states


def run_training_loop():
    """
    Runs the variational optimization loop using torchquantum.
    """
    print("==================================================")
    print(" torchquantum Legacy QNN Optimization Loop")
    print("==================================================")
    
    model = TorchQuantumQNN()
    print("Model initialized successfully.")
    print(f"Initial parameters: theta0 = {model.theta0.item():.4f}, theta1 = {model.theta1.item():.4f}")
    
    # Define target state: Bell State |Phi+> = 1/sqrt(2) * (|00> + |11>)
    target_state = torch.zeros((1, 4), dtype=torch.complex64)
    target_state[0, 0] = 1.0 / math.sqrt(2.0)
    target_state[0, 3] = 1.0 / math.sqrt(2.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    epochs = 40
    print(f"\nTraining QNN for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass (simulates circuit)
        output_state = model()
        
        # Calculate overlap and fidelity
        overlap = torch.sum(torch.conj(output_state) * target_state)
        fidelity = torch.abs(overlap) ** 2
        loss = 1.0 - fidelity
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}: Loss = {loss.item():.6f} | Fidelity = {fidelity.item():.6f}")
            
    print("\nOptimization Complete!")
    print(f"Final parameters: theta0 = {model.theta0.item():.4f}, theta1 = {model.theta1.item():.4f}")


def show_diagnostics():
    """
    Prints diagnostic information and setup guide for legacy torchquantum environments.
    """
    print("======================================================================")
    print(" WARNING: torchquantum is not available in the current environment")
    print("======================================================================")
    print("\n[Reason]")
    print("The 'torchquantum' library (MIT HAN Lab) was last updated in late 2022")
    print("and has a strict dependency on legacy pre-1.0 Qiskit packages (e.g. qiskit-aer == 0.11.0).")
    print("In modern environments running Python 3.12+ or Qiskit 2.x, these legacy packages")
    print("fail to compile, raising errors such as 'ModuleNotFoundError: No module named qiskit.providers.aer'.")
    print("\n[Solution to run this script]")
    print("To execute this specific torchquantum model, you must set up a legacy environment:")
    print("  1. Create a Python 3.10 or 3.11 virtual environment.")
    print("  2. Install legacy Qiskit: pip install \"qiskit==0.45.2\" \"qiskit-aer==0.12.2\"")
    print("  3. Install torchquantum without dependencies: pip install torchquantum --no-deps")
    print("  4. Install supporting libraries: pip install opt-einsum")
    print("\n[Alternative]")
    print("For a fully working, modern alternative running on Python 3.14+ without legacy dependencies,")
    print("run our first-principles simulator:")
    print("  python PyTorch-Quantum/01_pytorch_statevector.py")
    print("======================================================================")


def main():
    if TORCHQUANTUM_AVAILABLE:
        run_training_loop()
    else:
        show_diagnostics()


if __name__ == "__main__":
    main()
