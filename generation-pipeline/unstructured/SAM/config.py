from typing import Optional


_DEVICE = "cpu"


def configure_device(use_cuda: bool = False, gpu: Optional[int] = None):
    global _DEVICE
    if use_cuda:
        if gpu is not None:
            _DEVICE = f"cuda:{gpu}"
        else:
            _DEVICE = "cuda"
    else:
        _DEVICE = "cpu"


def get_device() -> str:
    global _DEVICE
    return _DEVICE
