import torch.nn as nn


def get_activation_module(activation_str):
    activation_str = activation_str.replace("_", "").lower()
    activation_str = activation_str.replace("()", "")
    activation_str = activation_str.strip()
    if activation_str == "relu":
        return nn.ReLU()
    elif activation_str == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_str}")
