import os

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)


def save_distributed(model: torch.nn.Module, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    dcp.save({"model": get_model_state_dict(model)}, checkpoint_id=directory)


def load_distributed(model: torch.nn.Module, directory: str) -> None:
    model_state = get_model_state_dict(model)
    dcp.load({"model": model_state}, checkpoint_id=directory)
    set_model_state_dict(model, model_state)
