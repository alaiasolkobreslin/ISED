import os

from .config import get_device

_MODEL_TYPE = "default"

_CHECKPOINT = None

_SAM = None


def configure_sam(args):
    global _CHECKPOINT
    try:
        if "sam_checkpoint" in args:
            _CHECKPOINT = args.sam_checkpoint
        else:
            _CHECKPOINT = os.getenv("SAM_CHECKPOINT")
        if _CHECKPOINT is None:
            return
    except:
        pass


def get_sam_model():
    global _MODEL_TYPE
    global _CHECKPOINT
    global _SAM

    if _SAM is None:
        from segment_anything import sam_model_registry
        _SAM = sam_model_registry[_MODEL_TYPE](
            checkpoint=_CHECKPOINT).to(device=get_device())
        return _SAM
    else:
        return _SAM
