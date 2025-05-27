import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from typing import Union, Tuple
import math
#code was taken from timm/data/transforms.py

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype)


class ToTensorNormalize:

    def __init__(self, mean, std, dtype=torch.float32, device=torch.device('cpu')):
        self.dtype = dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=device)
        std = torch.as_tensor(std, dtype=dtype, device=device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        self.mean = mean
        self.std = std

    def __call__(self, pil_img):
        mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
        img = torch.from_numpy(
            np.array(pil_img, mode_to_nptype.get(pil_img.mode, np.uint8))
        )
        if pil_img.mode == '1':
            img = 255 * img
        img = img.view(pil_img.size[1], pil_img.size[0], len(pil_img.getbands()))
        img = img.permute((2, 0, 1))
        if isinstance(img, torch.ByteTensor):
            img = img.to(self.dtype)
            img.sub_(self.mean * 255.).div_(self.std * 255.)
        else:
            img = img.to(self.dtype)
            img.sub_(self.mean).div_(self.std)
        return img


def transforms_imagenet_eval(
        img_size: Union[int, Tuple[int]] = 224,
        crop_pct=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        normalize=False,
        compose=True,
):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # FIXME handle case where img is square and we want non aspect preserving resize
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    preaugment_tfl = [transforms.Resize(scale_size, transforms.InterpolationMode(interpolation)),
                      transforms.CenterCrop(img_size),
                     ]
    preaugment = transforms.Compose(preaugment_tfl) if compose else preaugment_tfl

    if normalize:
        preprocess_tfl = [ToTensorNormalize(mean=mean, std=std)]
    else:
        # (pre)fetcher and collate will handle tensor conversion and normalize
        preprocess_tfl = [ToNumpy()]

    preprocess = transforms.Compose(preprocess_tfl) if compose else preprocess_tfl

    return preaugment, preprocess


