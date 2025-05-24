import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps, Image
from torchvision import transforms

## https://github.com/google-research/augmix
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
def _augmix_aug(x_orig,args):
    if args.normalize_data:
        print(f"normalizing data!!!! -> exit program")
        exit(1)
        preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
    else:
        preprocess = transforms.Compose([transforms.ToTensor()])
    preaugment = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
    x_orig = preaugment(x_orig)
    x_processed = preprocess(x_orig)
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))
    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(augmentations)(x_aug)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix

cifar10_aug = _augmix_aug


def autocontrast(pil_img, level=None):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, level=None):
    return ImageOps.equalize(pil_img)

def rotate(pil_img, level):
    degrees = int_parameter(rand_lvl(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)

def solarize(pil_img, level):
    level = int_parameter(rand_lvl(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

def shear_x(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def shear_y(pil_img, level):
    level = float_parameter(rand_lvl(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def translate_x(pil_img, level):
    level = int_parameter(rand_lvl(level), 32 / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

def translate_y(pil_img, level):
    level = int_parameter(rand_lvl(level), 32 / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((32, 32), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR, fillcolor=128)

def posterize(pil_img, level):
    level = int_parameter(rand_lvl(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.

def rand_lvl(n):
    return np.random.uniform(low=0.1, high=n)


augmentations = [
    autocontrast,
    equalize,
    lambda x: rotate(x, 1),
    lambda x: solarize(x, 1),
    lambda x: shear_x(x, 1),
    lambda x: shear_y(x, 1),
    lambda x: translate_x(x, 1),
    lambda x: translate_y(x, 1),
    lambda x: posterize(x, 1),
]


def compute_jacobian_correct_class(inputs, outputs, targets):
    batch_size = inputs.shape[0]
    output_dim = outputs.shape[-1]
    cotangents = torch.zeros(batch_size, output_dim, device=inputs.device)
    cotangents[range(batch_size), targets] = 1.0
    jacobian_row = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=cotangents,
        create_graph=False,
        retain_graph=False,
    )[0]
    return jacobian_row


def compute_gp_max_output(inputs, outputs):
    batch_size = inputs.shape[0]
    output_dim = outputs.shape[-1]
    max_indices = torch.argmax(outputs, dim=1)
    cotangents = torch.zeros(batch_size, output_dim, device=inputs.device)
    cotangents[range(batch_size), max_indices] = 1.0
    jacobian_row = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=cotangents,
        create_graph=False,
        retain_graph=False,
    )[0]
    return jacobian_row


def single_eval_test_jacobian_norm(args, model, device, image, target, jacobian_type, te_transforms):
    model.eval()
    inputs = te_transforms(image).unsqueeze(0).cuda()
    tardet_expanded = torch.tensor(target).repeat(inputs.shape[0]).to(inputs.device)
    test_loss = 0
    batch_size = 1
    data = inputs.to(device)
    target = tardet_expanded.to(device)
    print(f"####### process batch #######")
    data.requires_grad = True
    output = model(data)
    print(f"DEBUG: eval_test_jacobian_norm: data.shape={data.shape} output.shape={output.shape}")
    test_loss += F.cross_entropy(output, target, size_average=False).item()
    # Compute Jacobian for model
    if "gp_correct_class" in jacobian_type:
        print(f"jacobian_type=gp_correct_class")
        jacobian_row = compute_jacobian_correct_class(data, output, target)
    else:
        print(f"jacobian_type=gp_max_output")
        jacobian_row = compute_gp_max_output(data, output)
    print(f"jacobian_row.shape={jacobian_row.shape}")
    # Compute L1, L2, and Linf norms
    jacobian_l1_norm = jacobian_row.view(batch_size, -1).norm(1, 1) # (p=1,dim=1)
    jacobian_l2_norm = jacobian_row.view(batch_size, -1).norm(2, 1)
    jacobian_linf_norm = jacobian_row.view(batch_size, -1).abs().max(dim=1)[0]  # L-infinity norm
    return jacobian_l1_norm, jacobian_l2_norm, jacobian_linf_norm

def single_eval_test_jacobian_norm_zero_aug(args, model, device, data, target, jacobian_type):
    model.eval()
    test_loss = 0
    batch_size = 1
    print(f"####### process batch #######")
    data.requires_grad = True
    output = model(data)
    print(f"DEBUG: eval_test_jacobian_norm: data.shape={data.shape} output.shape={output.shape}")
    test_loss += F.cross_entropy(output, target, size_average=False).item()
    # Compute Jacobian for model
    if "gp_correct_class" in jacobian_type:
        print(f"jacobian_type=gp_correct_class")
        jacobian_row = compute_jacobian_correct_class(data, output, target)
    else:
        print(f"jacobian_type=gp_max_output")
        jacobian_row = compute_gp_max_output(data, output)
    print(f"jacobian_row.shape={jacobian_row.shape}")
    # Compute L1, L2, and Linf norms
    jacobian_l1_norm = jacobian_row.view(batch_size, -1).norm(1, 1) # (p=1,dim=1)
    jacobian_l2_norm = jacobian_row.view(batch_size, -1).norm(2, 1)
    jacobian_linf_norm = jacobian_row.view(batch_size, -1).abs().max(dim=1)[0]  # L-infinity norm
    return jacobian_l1_norm, jacobian_l2_norm, jacobian_linf_norm