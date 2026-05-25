from dataclasses import fields
import importlib
import inspect
import os
import shutil
from typing import Any, Dict, List, Optional

import yaml

def load_model_class(identifier: str, prefix: str = "models."):

    module_path, class_name = identifier.split("@")
    module = importlib.import_module(prefix + module_path)
    return getattr(module, class_name)

def get_model_source_path(identifier: str, prefix: str = "models.") -> Optional[str]:

    module_path, _ = identifier.split("@")
    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)

def dict_view(obj: Any) -> Dict[str, Any]:

    return {f.name: getattr(obj, f.name) for f in fields(obj)}

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
        module = importlib.import_module(module_name)
        source_file = inspect.getsourcefile(module)
        if source_file:
            dest = os.path.join(code_dir, module_name.replace(".", "__") + ".py")
            shutil.copy(source_file, dest)
