import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Applying ReLU activation function to the hidden layer
        x = self.fc2(x)          # Output layer (no activation function for regression tasks)
        return x

# Example usage:
input_size = 10
hidden_size = 20
output_size = 1  # For simplicity, assuming a single output node for regression task
model = SimpleNN(input_size, hidden_size, output_size)

# Generating random input tensor for demonstration
input_tensor = torch.randn(32, input_size)  # batch_size=32
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Shape should be (32, output_size)
