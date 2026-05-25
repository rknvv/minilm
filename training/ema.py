from contextlib import contextmanager
from typing import Dict

import torch


def _local(t: torch.Tensor) -> torch.Tensor:
    return getattr(t, "to_local")() if hasattr(t, "to_local") else t


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: _local(param).detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is not None:
                shadow.lerp_(_local(param).detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is not None:
                _local(param).copy_(shadow)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        backup = {
            name: _local(param).detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }
        self.copy_to(model)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup:
                        _local(param).copy_(backup[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]
