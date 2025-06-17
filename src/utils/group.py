import torch

class SE2Group:
    def __init__(self, x, y, theta):
        self.x = x.clone()
        self.y = y.clone()
        self.theta = theta.clone()

    def __add__(self, other):
        x = self.x + other.x * torch.cos(self.theta) - other.y * torch.sin(self.theta)
        y = self.y + other.y * torch.cos(self.theta) + other.x * torch.sin(self.theta)
        theta = self.theta + other.theta
        theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi
        return SE2Group(x, y, theta)

    def parameters(self):
        self.theta = (self.theta + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.stack([self.x, self.y, self.theta], dim=-1)

    @classmethod
    def from_parameters(cls, parameters):
        x, y, theta = torch.split(parameters, 1, dim=-1)
        return cls(x.squeeze(-1), y.squeeze(-1), theta.squeeze(-1))

    @classmethod
    def from_batched_parameters(cls, parameters):
        if parameters.ndim == 2 and parameters.shape[-1] == 3:
            x, y, theta = torch.split(parameters, 1, dim=-1)
            return cls(x.squeeze(-1), y.squeeze(-1), theta.squeeze(-1))
        else:
            raise ValueError("Input parameters must have shape [batch_size, 3].")