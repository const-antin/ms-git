import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) # NN linear layer
        self.fc2 = nn.Linear(hidden_size, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Applying ReLU activation function to the hidden layer
        x = F.relu(self.fc2(x))  # Applying second hidden layer
        x = self.fc3(x)          # Output layer (no activation function for regression tasks)
        return x

# Example usage:
input_size = 15
hidden_size = 20
hidden_size_2 = 30
output_size = 1  # For simplicity, assuming a single output node for regression task
model = SimpleNN(input_size, hidden_size, hidden_size_2, output_size)

# Generating random input tensor for demonstration
input_tensor = torch.randn(32, input_size)  # batch_size=32
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Shape should be (32, output_size)
