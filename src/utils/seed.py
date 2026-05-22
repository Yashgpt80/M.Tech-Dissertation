import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN is enabled by default (torch 2.5.1+cu121 is stable). Set
    # YAAS_USE_CUDNN=0 to disable if you hit version mismatch errors.
    if os.environ.get("YAAS_USE_CUDNN", "1") == "0":
        torch.backends.cudnn.enabled = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
