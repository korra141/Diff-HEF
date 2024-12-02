import torch
def check_model_weights_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in model weights: {name}")

def get_gradients(model):
    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append(param.grad.norm().item())
    return gradients

# prompt: Write a function to check if the tensor has nan values and print statements

def check_tensor_nan(tensor):
    if torch.isnan(tensor).any():
        print("Tensor contains NaN values.")