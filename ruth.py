import torch
from torch import nn


# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded


# Define data (using the provided example)
data = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0]
], dtype=torch.float32)

# Define some parameters (experiment with different encoding dimensions)
input_dim = data.shape[1]  # Get the number of features from data
encoding_dim = 3  # You can adjust this value based on your desired compression

# Create the Autoencoder model
model = Autoencoder(input_dim, encoding_dim)

# Define loss function and optimizer (replace with suitable ones for your task)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop (modify for your training epochs)
for epoch in range(200000):
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, data)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss (optional)

    if (epoch + 1) % 1000 == 0:
        print(f"Iteration:{epoch},Epoch: {epoch + 1}/{20000}, Loss: {loss.item():.4f}")

# After training, use the model for encoding and decoding

encoded_data = model.encoder(data)
decoded_data = model.decoder(encoded_data)

# Print the original, encoded, and decoded data
print("Original Data:")
print(data)
print("\nEncoded Data:")
print(encoded_data)
print("\nDecoded Data:")
print(decoded_data)
