import torch
import logging
import logging.handlers

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
        
def setup_logger(queue):
    """
    Setup logger for a specific process.
    Args:
        queue (mp.Queue): The queue used to send log messages to the main process.
        rank (int): The rank or ID of the current process.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a QueueHandler to send logs to the multiprocessing queue
    queue_handler = logging.handlers.QueueHandler(queue)
    logger.addHandler(queue_handler)

    return logger
