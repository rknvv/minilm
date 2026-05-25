import logging
from typing import Union

import torch

try:
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
except ImportError:
    from torch.distributed._composable.fsdp import (
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )

from models.transformer import TransformerBlock

logger = logging.getLogger(__name__)

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_fsdp_mesh(world_size: int):
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))


def apply_fsdp2(
    model: torch.nn.Module,
    dtype: Union[str, torch.dtype],
    mesh,
    reshard_after_forward: bool = False,
) -> torch.nn.Module:
    param_dtype = _DTYPES.get(dtype, torch.bfloat16) if isinstance(dtype, str) else dtype
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=torch.float32
    )

    for module in model.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(
                module,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(
        model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward
    )
    assert isinstance(model, FSDPModule)
    return model
