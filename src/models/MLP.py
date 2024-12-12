import torch
import torch.nn as nn
import math
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, grid_size,batch_size):
        super(NeuralNetwork, self).__init__()
        self.batch_size = batch_size
        self.band_limit_x, self.band_limit_y = grid_size
        output_dim = self.band_limit_x * self.band_limit_y
        self.layer1 = nn.Linear(input_dim, 200)
        self.layer2 = nn.Linear(200,300)
        self.layer3 = nn.Linear(300, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input_energy):
        input_energy_reshaped = input_energy.view(self.batch_size,-1)
        # print(input_energy_reshaped.shape)
        x = self.sigmoid(self.layer1(input_energy_reshaped))
        x = self.sigmoid(self.layer2(x))
        x = self.layer3(x)
        x = self.relu(x)
        x = x.view(-1, self.band_limit_x, self.band_limit_y)
        return torch.clamp(-x,min=-10) + input_energy

def init_weights_mlp(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

def init_weights_one(m):
        if isinstance(m, nn.Linear):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        self._initialize_weights()

    def forward(self, x):
        y_pred = self.model(x)
        mu_x = y_pred[:,0:1] %(2 * math.pi)
        mu_y = y_pred[:,1:2] %(2 * math.pi)
        logcov_x = y_pred[:,2:3]
        logcov_y = y_pred[:,3:4]
        epsilon = torch.tensor(-1e-2)  # Small value to avoid zero covariance
        logcov_x = torch.where(logcov_x < epsilon, epsilon, logcov_x)
        logcov_y = torch.where(logcov_y < epsilon, epsilon, logcov_y)
        return mu_x, mu_y, logcov_x, logcov_y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)