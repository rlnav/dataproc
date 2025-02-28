import torch
from monoforce.models.traj_predictor.traj_lstm import TrajLSTM


# Example usage
if __name__ == "__main__":
    # Parameters
    state_dim = 6
    height_channels = 1  # e.g., single heightmap (grayscale)
    control_dim = 2  # e.g., linear and angular velocities
    lstm_hidden_size = 128
    lstm_layers = 1
    seq_len = 500  # Number of time steps
    batch_size = 8
    height, width = 128, 128  # Size of input heightmap

    # Model
    model = TrajLSTM(state_dim, height_channels, control_dim, lstm_hidden_size, lstm_layers)

    # Example input data
    state0 = torch.randn(batch_size, state_dim)  # Initial state
    heightmap = torch.randn(batch_size, height_channels, height, width)  # Shared heightmap
    controls = torch.randn(batch_size, seq_len, control_dim)  # Sequence of control inputs

    # Forward pass
    states = model(state0, heightmap, controls)
    print("Predictions shape:", states.shape)
