import torch
import pdb
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.metrics import absolute_error_s1, error_s1

class ExtendedKalmanFilter:
    def __init__(self,initial_covariance, process_cov):
        self.P = torch.tensor(initial_covariance) # Covariance matrix
        self.Q = torch.tensor(process_cov)  # Process noise covariance
    
    def set_initial_state(self, x):
        """
        Set the initial state of the filter.
        x: Initial state
        """
        self.x = x
    
    def set_step(self, step):
        """
        Set the step of the filter.
        step: Step of the filter
        """
        self.step = step

    def predict(self):
        """
        Prediction step.
        f: Non-linear state transition function
        F: Jacobian of the state transition function
        u: Control input (optional)
        """
        positive_angle = torch.ones_like(self.x) * (2*math.pi)
        self.x = (self.x + self.step) % (2*math.pi)
        if torch.any(self.x < 0):
            pdb.set_trace()
        self.P =  self.P + self.Q
    
    # def check_left_or_right(self,z,z_pred):
    #     check = z - z_pred
    #     left = torch.where(check < 0 and check.abs() < math.pi , True, False)
    #     return left

    def update(self, z, R):
        """
        Update step of EKF.
        :param z: Measurement (numpy array)
        :param H: Jacobian of measurement model (numpy array)
        :param h: Nonlinear measurement function (callable, h(x))
        """
        z_pred = self.x
        # jacobian_h = torch.ones_like(z_pred)
        positive_angle = torch.ones_like(z_pred) * (2*math.pi)
        # Innovation (residual)s
        y = error_s1(z,z_pred)
        # Innovation covariancet
        S = self.P + R

        # Kalman gain
        K = self.P  / S
        
        # Update state estimate
        self.x = (self.x + K * y) % (2*math.pi) # Update state estimate
        if torch.any(self.x < 0):
            pdb.set_trace()
        # Update covariance estimate
        identity = torch.ones(K.shape)
        self.P = (identity - K) * self.P
