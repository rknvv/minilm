import importlib
import inspect
import logging
import os
import shutil
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

def snapshot_run(
    out_dir: str,
    config_sections: Dict[str, Any],
    code_modules: List[str],
) -> None:

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_config.yaml"), "w") as f:
        yaml.safe_dump(config_sections, f, sort_keys=False, allow_unicode=True)

    code_dir = os.path.join(out_dir, "code_snapshot")
    os.makedirs(code_dir, exist_ok=True)
    for module_name in code_modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            # e.g. models.fused_loss needs triton+liger, absent on CPU-only boxes.
            logger.warning("Skipping code snapshot of %s: %s", module_name, exc)
            continue
        source_file = inspect.getsourcefile(module)
        if source_file:
            dest = os.path.join(code_dir, module_name.replace(".", "__") + ".py")
            shutil.copy(source_file, dest)
