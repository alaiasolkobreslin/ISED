from typing import *
import os

from . import sam_config

import torch
import numpy as np
import cv2

DUMP_IMAGE_PATH = ".tmp/scallop-sam/"


def segment_anything(
    # item,
    # fields: List[str],
    # *,
    img: torch.Tensor,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    dump_image: bool = True,
    iou_threshold: float = 0.88,
    area_threshold: int = 1000,
    # limit: Optional[int] = None,
    expand_crop_region: int = 0,
    debug: bool = False,
):

    sam_config.configure_sam([])
    sam = sam_config.get_sam_model()
    if sam is not None:
        from segment_anything import SamPredictor
        predictor = SamPredictor(sam)

        img = torch.transpose(input=img, dim0=0, dim1=1)
        img = torch.transpose(input=img, dim0=1, dim1=2)
        cvt_img = torch.stack(
            [img[:, :, 2], img[:, :, 1], img[:, :, 0]], dim=2)  # BGR to RGB
        np_image = cvt_img.numpy()
        predictor.set_image(np_image)

        input_box = np.array([xmin, ymin, xmax, ymax])

        masks, iou_preds, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=None,
        )

        masks_data = torch.from_numpy(masks)
        rle = mask_to_rle_pytorch(masks_data)[0]
        areas = area_from_rle(rle)

        # print(f"bbox area: {areas}")

        mask = torch.from_numpy(masks[0])
        if dump_image:
            save_temporary_image(id(img), 0, "mask", mask)

        yield (areas, mask)


def mask_to_rle_pytorch(tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    import torch

    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype,
                             device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype,
                             device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])


def save_temporary_image(img_id, i, kind, img_tensor):
    from PIL import Image

    # First get the directory
    directory = os.path.join(DUMP_IMAGE_PATH, f"{img_id}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Dump the image
    file_name = os.path.join(directory, f"{kind}-{i}.jpg")
    img = Image.fromarray(img_tensor.numpy())
    img.save(open(file_name, "w"))
