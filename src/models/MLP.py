import torch
import torch.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, grid_size,batch_size):
        super(NeuralNetwork, self).__init__()
        self.batch_size = batch_size
        self.band_limit_x, self.band_limit_y = grid_size
        output_dim = self.band_limit_x * self.band_limit_y
        self.layer1 = nn.Linear(input_dim, 1000)
        self.layer2 = nn.Linear(1000,2000)
        self.layer3 = nn.Linear(2000, output_dim)
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
        return torch.clamp(x,min=1e-8,max=50)

def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

def init_weights_one(m):
        if isinstance(m, nn.Linear):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)