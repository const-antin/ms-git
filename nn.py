import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()
        self.fc = []

        sizes = [input_size] + hidden_sizes + [output_size]

        for in_size, out_size in zip(sizes[0:], sizes[1:]):
            self.fc.append(nn.Linear(in_size, out_size))
        
    def forward(self, x):
        for fc in self.fc[:-1]:
            x = F.relu(fc(x))       # All inner layers have an activation
        x = self.fc[-1](x)          # Output layer (no activation function for regression tasks)
        return x

# Example usage:
input_size = 10
hidden_sizes = [20, 40, 50]
output_size = 1  # For simplicity, assuming a single output node for regression task
model = SimpleNN(input_size, hidden_sizes, output_size)

# Generating random input tensor for demonstration
input_tensor = torch.randn(32, input_size)  # batch_size=32
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Shape should be (32, output_size)
