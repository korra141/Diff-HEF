import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DensityPredictorCNN(nn.Module):
    def __init__(self, input_dim, grid_size,batch_size):
        """
        Args:
            input_dim: Number of input features.
            grid_size: Tuple (H, W) representing the dimensions of the grid.
        """
        super(DensityPredictorCNN, self).__init__()
        self.grid_size = grid_size

        self.input_padding = nn.ReplicationPad2d(1)

        # Convolutional layers for refinement
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=3,padding=1)
        self.batch_size = batch_size

    def forward(self, input):

        # Convolutional layers

        x = input.unsqueeze(1)
        x = self.input_padding(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(x.squeeze(1))
        # batch_size, dim1, dim2, *rest = x.shape
        # x = x.view(batch_size, dim1 * dim2, *rest)

        # # # Apply softmax along the combined dimension (dim1 * dim2)
        # x = F.softmax(x, dim=1)

        # # # Reshape back to the original shape
        # x = x.view(batch_size, dim1, dim2, *rest)

        x = x[:,1:1+self.grid_size[0], 1:1+self.grid_size[1]]

        return torch.clamp(x,min=1e-10)
    
class LikelihoodPredicitionCNN(nn.Module):
    def __init__(self, grid_size,batch_size):
        super(LikelihoodPredicitionCNN, self).__init__()
        self.grid_size = grid_size

        self.input_padding = nn.ReplicationPad2d(1)

        # Convolutional layers for refinement
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3,padding=1)
        self.batch_size = batch_size

    def forward(self,density_1,density_2):
        density_1 = density_1.unsqueeze(1)
        density_2 = density_2.unsqueeze(1)
        x = torch.cat((density_1, density_2), dim=1) 
        x = self.input_padding(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(x.squeeze(1))
        x = x[:,1:1+self.grid_size[0], 1:1+self.grid_size[1]]
        return torch.clamp(x,min=1e-10)
    

class CNNModel_A(nn.Module):
    def __init__(self, grid_size):
        super(CNNModel_A, self).__init__()
        self.grid_size = grid_size

        # Define layers
        self.input_padding = nn.ReflectionPad2d(1)  # Example padding
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(4)  # BatchNorm for conv1
        self.conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(1)  # BatchNorm for conv2


    def forward(self, input):
        # Apply layers with BatchNorm and activation
        x = input.unsqueeze(1)
        x = self.input_padding(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))  # Conv1 + BN + Activation
        x = F.relu(self.conv2(x).squeeze(1))  # Conv2 + BN + Activation
        # x = self.conv3(x)  # Conv3
        # x = self.bn3(x)  # Optional BatchNorm for last layer
        # x = F.relu(x.squeeze(1))  # Squeeze and ReLU
        x = x[:, 1:1 + self.grid_size[0], 1:1 + self.grid_size[1]]

        # Clamp output
        return torch.clamp(x, min=1e-10)

        
class DensityPredictorMLPCNN(nn.Module):
    def __init__(self, input_dim, grid_size,batch_size):
        """
        Args:
            input_dim: Number of input features.
            grid_size: Tuple (H, W) representing the dimensions of the grid.
        """
        super(DensityPredictorMLPCNN, self).__init__()
        self.grid_size = grid_size

        self.input_padding = nn.ReplicationPad2d(1)
        self.linear1 = nn.Linear(input_dim, math.prod(grid_size)//2)
        self.linear2 = nn.Linear(math.prod(grid_size)//2, math.prod(grid_size))

        # Convolutional layers for refinement
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3,padding=1)
        # self.conv2 = nn.Conv2d(4, 4, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3,padding=1)
        self.batch_size = batch_size

    def forward(self, input):

        # Convolutional layers
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2(x))
        x = x.view(self.batch_size, 1, *self.grid_size)
        x = self.input_padding(x)
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(x.squeeze(1))
        x = x[:,1:1+self.grid_size[0], 1:1+self.grid_size[1]]

        return torch.clamp(x,min=1e-10)

def init_weights(model):
  for layer in model.modules():
    initialize_conv_to_not_zero(layer)

def initialize_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def init_weights_identity(model):
  for layer in model.modules():
    initialize_conv_to_identity(layer)

def init_weights_zero(model):
  for layer in model.modules():
    initialize_conv_to_zero(layer)

def init_weights(model):
    for layer in model.modules():
        initialize_conv_to_not_zero(layer)


def initialize_conv_to_zero(conv_layer):
    """
    Initialize a convolutional layer such that its output is close to zero.
    Args:
        conv_layer (nn.Conv2d): The convolutional layer to initialize.
    """
    if isinstance(conv_layer, nn.Conv2d):
      # Small random initialization for weights
      # nn.init.normal_(conv_layer.weight, mean=0.0, std=1e-3)
      nn.init.kaiming_uniform_(conv_layer.weight,mode='fan_in', nonlinearity='relu')
      nn.init.constant_(conv_layer.bias, 0.0)

def initialize_conv_to_not_zero(conv_layer,noise_scale=0.01):
    """
    Initialize a convolutional layer such that its output is close to zero.
    Args:
        conv_layer (nn.Conv2d): The convolutional layer to initialize.
    """
    if isinstance(conv_layer, nn.Conv2d):
      # Small random initialization for weights
      nn.init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
      noise = torch.randn_like(conv_layer.weight) * noise_scale
      conv_layer.weight.data += noise
    #   nn.init.kaiming_normal_(conv_layer.weight,mode='fan_in', nonlinearity='leaky_relu')
      nn.init.constant_(conv_layer.bias, 0.0)



def initialize_identity_with_torch(conv_layer):
    """
    Initialize a Conv2d layer as an identity transformation using PyTorch utilities.
    """
    if isinstance(conv_layer, nn.Conv2d):
      with torch.no_grad():
          # Get kernel size and ensure it's odd
          kernel_size = conv_layer.kernel_size[0]
          assert kernel_size % 2 == 1, "Kernel size must be odd for identity initialization."

          # Get the number of input and output channels
          out_channels, in_channels, h, w = conv_layer.weight.shape

          # Create an identity kernel
          identity_kernel = torch.zeros_like(conv_layer.weight)
          for i in range(min(out_channels, in_channels)):
              identity_kernel[i, i, kernel_size // 2, kernel_size // 2] = 1.0

          # Assign the identity kernel to the weights
          conv_layer.weight.copy_(identity_kernel)

          # Initialize bias to zero
          if conv_layer.bias is not None:
              nn.init.zeros_(conv_layer.bias)

def initialize_conv_to_identity(conv_layer):
    """
    Initialize a Conv2D layer to act as an identity mapping (mirroring the input).
    Args:
        conv_layer (nn.Conv2d): The Conv2D layer to initialize.
    """
    if isinstance(conv_layer, nn.Conv2d):
      with torch.no_grad():
        if conv_layer.weight.size(2) % 2 == 0 or conv_layer.weight.size(3) % 2 == 0:
            raise ValueError("Kernel size must be odd to achieve perfect identity mapping.")

        wts = torch.zeros(1, 1, 3, 3)
        nn.init.dirac_(wts)
        conv_layer.weight.copy_(wts)

        # Set biases to zero
        if conv_layer.bias is not None:
            nn.init.constant_(conv_layer.bias, 0.0)
