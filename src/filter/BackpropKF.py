import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

class BackpropKalmanFilter(nn.Module):
    def __init__(self, state_dim, measurement_dim):
        super(BackpropKalmanFilter, self).__init__()
        
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # State transition and measurement matrices (learnable)
        self.F = nn.Parameter(torch.eye(state_dim))  # State transition matrix
        self.H = nn.Parameter(torch.eye(measurement_dim, state_dim))  # Measurement matrix
        
        # Covariance matrices (learnable)
        self.Q = nn.Parameter(torch.eye(state_dim))  # Process noise covariance
        self.R = nn.Parameter(torch.eye(measurement_dim))  # Measurement noise covariance
        
        # Initialize state and covariance
        self.x = torch.zeros(state_dim, 1)  # State vector
        self.P = torch.eye(state_dim)       # Covariance matrix

    def predict(self):
        """
        Prediction step of the Kalman Filter.
        """
        # Predict the state
        self.x = self.F @ self.x
        
        # Predict the covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Update step of the Kalman Filter.
        z: Measurement vector
        """
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ torch.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + K @ y
        
        # Update covariance
        I = torch.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

    def forward(self, measurements):
        """
        Forward pass through the Kalman Filter for a sequence of measurements.
        measurements: A sequence of measurements (time steps x measurement_dim)
        """
        estimated_states = []
        
        for z in measurements:
            self.predict()
            self.update(z)
            estimated_states.append(self.x.clone())
        
        return torch.stack(estimated_states, dim=0)

# Example Usage
if __name__ == "__main__":
    # Simulate a simple 1D system
    state_dim = 2
    measurement_dim = 1
    
    # Initialize Backprop Kalman Filter
    bkf = BackpropKalmanFilter(state_dim, measurement_dim)
    
    # Generate synthetic data
    timesteps = 10
    true_states = torch.zeros(timesteps, state_dim, 1)
    measurements = torch.zeros(timesteps, measurement_dim, 1)
    
    # Assume constant velocity model
    for t in range(1, timesteps):
        true_states[t] = torch.tensor([[1.0], [0.5]]) + true_states[t - 1]  # State update
        measurements[t] = bkf.H @ true_states[t] + torch.randn(measurement_dim, 1) * 0.1  # Add noise

    # Pass the measurements through the Backprop KF
    estimated_states = bkf(measurements)

    print("Estimated states:")
    print(estimated_states)
