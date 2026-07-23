import torch


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(to_device(value, device) for value in data)
    return data
