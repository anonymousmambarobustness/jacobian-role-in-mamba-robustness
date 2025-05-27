#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script
for evaluating pretrained models or training checkpoints against ImageNet or
similarly organized image datasets. It prioritizes canonical PyTorch, standard
Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman), and edited by
Edoardo Debenedetti (https://github.com/dedeswim) to work with Adversarial Training.

The original validation script can be found at
https://github.com/rwightman/pytorch-image-models/, and the original license at
https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE.
"""
import time
import argparse
import csv
import glob
import logging
import math
import os
import numpy as np
from collections import OrderedDict
from typing import Dict
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import yaml
from timm.bits import (AccuracyTopK, AvgTensor, Monitor, Tracker, initialize_device)
# from timm_bits.metric_accuracy import AccuracyTopK
# from timm_bits.avg_tensor import AvgTensor
# from timm_bits.monitor import Monitor
# from timm_bits.tracker import Tracker
# from timm_bits.device_env_factory import initialize_device
from timm.data import (RealLabelsImagenet, create_dataset, create_loader_v2, create_transform_v2, fetcher, resolve_data_config)
from timm.models import (apply_test_time_pool, create_model, is_model, list_models, load_checkpoint, xcit)
from timm.utils import natural_key, setup_default_logging
from torchvision import transforms

import src.attacks as attacks
#import src.models as models  # Import needed to register the extra models that are not in timm
import src.vmamba_models as models # Import needed to register the extra models that are not in timm
from src.random import random_seed
import src.utils as utils
from src.engine import regularization_term
from memo_cifar10_third_party import aug as cifar10_aug
from memo_cifar10_third_party import single_eval_test_jacobian_norm
from memo_imagenet_third_party import aug as imagenet_aug


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--num_classes', default=1000, type=int)
parser.add_argument('--dataset',
                    '-d',
                    metavar='NAME',
                    default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split',
                    metavar='NAME',
                    default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download',
                    action='store_true',
                    default=False,
                    help='Allow download of dataset for torch/ '
                    'and tfds/ datasets that support it.')
parser.add_argument('--model',
                    '-m',
                    metavar='NAME',
                    default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--img-size',
                    default=None,
                    type=int,
                    metavar='N',
                    help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size',
                    default=None,
                    nargs=3,
                    type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), '
                    'uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float, metavar='N', help='Input image center crop pct')
parser.add_argument('--mean',
                    type=float,
                    nargs='+',
                    default=None,
                    metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std',
                    type=float,
                    nargs='+',
                    default=None,
                    metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation',
                    default='',
                    type=str,
                    metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-nn',
                    '--no-normalize',
                    action='store_true',
                    default=False,
                    help='Avoids normalizing inputs (but it scales them in [0, 1]')
parser.add_argument('--normalize-model',
                    action='store_true',
                    default=False,
                    help='Performs normalization as part of the model')
parser.add_argument('--num-classes', type=int, default=None, help='Number classes in dataset')
parser.add_argument('--class-map',
                    default='',
                    type=str,
                    metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp',
                    default=None,
                    type=str,
                    metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq',
                    default=1,
                    type=int,
                    metavar='N',
                    help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
# parser.add_argument('--num-gpu', type=int, default=1,
#                     help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true', help='enable test time pool')
parser.add_argument('--pin-mem',
                    action='store_true',
                    default=False,
                    help='Pin CPU memory in DataLoader for more'
                    'efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last',
                    action='store_true',
                    default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp',
                    action='store_true',
                    default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--tf-preprocessing',
                    action='store_true',
                    default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema',
                    dest='use_ema',
                    action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript',
                    dest='torchscript',
                    action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results-file',
                    default='',
                    type=str,
                    metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels',
                    default='',
                    type=str,
                    metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels',
                    default='',
                    type=str,
                    metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--force-cpu',
                    action='store_true',
                    default=False,
                    help='Force CPU to be used even if HW accelerator exists.')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')

parser.add_argument('--attack',
                    default='',
                    type=str,
                    metavar='ATTACK',
                    help='What attack to use (default: "pgd")')
parser.add_argument('--attack-eps',
                    default=4,
                    type=float,
                    metavar='EPS',
                    help='The epsilon to use for the attack (default 4/255)')
parser.add_argument('--attack-lr',
                    default=None,
                    type=float,
                    metavar='ATTACK_LR',
                    help='Learning rate for the attack (default 1e-4)')
parser.add_argument('--attack-steps',
                    default=10,
                    type=int,
                    metavar='ATTACK_STEPS',
                    help='Number of steps to run attack for (default 10)')
parser.add_argument('--attack-norm',
                    default='linf',
                    type=str,
                    metavar='NORM',
                    help='The norm to use for the attack (default linf)')
parser.add_argument('--attack-boundaries',
                    default=(0, 1),
                    nargs=2,
                    type=int,
                    metavar='L H',
                    help='Boundaries of projection')
parser.add_argument('--log-wandb',
                    action='store_true',
                    default=False,
                    help='Log results to wandb using the run stored in the bucket')
parser.add_argument('--use-mp-loader', action='store_true', default=False, help='Use Torch XLA\'s  MP Loader')
parser.add_argument('--num-examples',
                    default=None,
                    type=int,
                    metavar='EXAMPLES',
                    help='Number of examples to use for the evaluation (default the entire dataset)')
parser.add_argument('--patch-size', default=None, type=int, metavar='N', help='The patch size to use')
parser.add_argument('--verbose', action='store_true', default=False, help='Runs autoattack in verbose mode')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--num_augs', default=32, type=int)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--reg_type', default="none", type=str)
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--pretraining_type', default="Nat", type=str)




def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def adapt_single(args,model,aug_func,optimizer,image,label,cfg):
    model.eval()
    for iteration in range(args.niter):
        inputs = [aug_func(x_orig=image,args=args,cfg=cfg) for _ in range(args.num_augs)]
        inputs = torch.stack(inputs).cuda()
        optimizer.zero_grad()
        if args.reg_type in ["gp","gp_correct_class", "gp_correct_class_l1", "gp_correct_class_l2", "gp_max_output"]:
            inputs.requires_grad = True

        outputs = model(inputs)
        loss, logits = marginal_entropy(outputs)
        if args.reg_type in ["gp", "gp_correct_class", "gp_correct_class_l1", "gp_correct_class_l2", "gp_max_output"]:
            labels_expanded = torch.tensor(label).repeat(args.num_augs)  # Assumes label is a single tensor scalar
            reg_val = regularization_term(inputs=inputs, outputs=outputs, targets=labels_expanded, regularization_type=args.reg_type, model=model)
            original_loss = loss.item()
            loss = loss + (args.alpha * reg_val)
            inputs.requires_grad = False
            print(f"iteration{iteration}: loss={loss} original_loss={original_loss} args.alpha*reg_val={args.alpha * reg_val} reg_val={reg_val}  alpha={args.alpha}")
        else:
            print(f"iteration{iteration}: loss={loss} ")
        print(f"loss.requires_grad={loss.requires_grad} inputs.requires_grad={inputs.requires_grad} outputs.requires_grad={outputs.requires_grad}")
        loss.backward()
        optimizer.step()


def get_adv_output(attack,model,inputs,target,dev_env,batch_stats_backup):
    if attack is not None:
        if dev_env.type_xla:
            model.train()
        with torch.enable_grad():
            adv_sample = attack(model, inputs, target)
        if dev_env.type_xla:
            utils.restore_batchnorm_stats(model, batch_stats_backup)
            model.eval()
        adv_output = model(adv_sample)
    else:
        adv_output = None

    return adv_output

def calc_clean_and_adv_accuracy(model,sample,target,te_transforms,accuracy,pgd_adv_accuracy,AA_adv_accuracy,real_labels,valid_labels,pgd_attack,AA_attack,dev_env,batch_stats_backup):
    model.eval()
    inputs = te_transforms(sample).unsqueeze(0).cuda()
    target = torch.tensor(target).repeat(inputs.shape[0]).to(inputs.device)
    with torch.no_grad():
        with dev_env.autocast():
            pgd_adv_output = get_adv_output(pgd_attack,model,inputs,target,dev_env,batch_stats_backup)
            AA_adv_output = get_adv_output(AA_attack,model,inputs,target,dev_env,batch_stats_backup)
            output = model(inputs)
        if valid_labels is not None:
            output = output[:, valid_labels]

        if dev_env.type_xla:
            dev_env.mark_step()
        elif dev_env.type_cuda:
            dev_env.synchronize()

        if real_labels is not None:
            real_labels.add_result(output)
        accuracy.update(output.detach(), target)
        if pgd_adv_output is not None:
            pgd_adv_accuracy.update(pgd_adv_output.detach(), target)
        if AA_adv_output is not None:
            AA_adv_accuracy.update(AA_adv_output.detach(), target)

def get_initial_model_and_optimizer(args,model,dev_env):
    if model is None:
        model = create_model(args.model,
                             pretrained=args.pretrained,
                             num_classes=args.num_classes,
                             in_chans=3,
                             global_pool=args.gp,
                             scriptable=args.torchscript)
        passed_model_none = True
    else:
        passed_model_none = False

    if args.num_classes is None:
        assert hasattr(model,'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint.startswith('gs://') and passed_model_none:
        model = utils.load_model_from_gcs(args.checkpoint,
                                          args.model,
                                          pretrained=args.pretrained,
                                          num_classes=args.num_classes,
                                          in_chans=3,
                                          global_pool=args.gp,
                                          scriptable=args.torchscript)

    elif args.checkpoint and passed_model_none:
        load_checkpoint(model, args.checkpoint, args.use_ema)
    else:
        assert model is not None

    if args.patch_size is not None and isinstance(model, xcit.XCiT) and model.patch_embed.patch_size != args.patch_size:
        assert args.patch_size in {2, 4, 8}, "Finetuning patch size can be only 4, 8 or `None`"
        assert isinstance(model, models.xcit.XCiT), "Finetuning patch size is only supported for XCiT"
        model.patch_embed.patch_size = args.patch_size
        model.patch_embed.proj[0][0].stride = (1, 1)
        if args.patch_size == 4:
            model.patch_embed.proj[2][0].stride = (1, 1)
        if args.patch_size == 2:
            model.patch_embed.proj[4][0].stride = (1, 1)

    param_count = sum([m.numel() for m in model.parameters()])

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)
    data_config['normalize'] = not (args.no_normalize or args.normalize_model)

    if args.normalize_model:
        mean = args.mean or data_config["mean"]
        std = args.std or data_config["std"]
        model = utils.normalize_model(model, mean=mean, std=std)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    # FIXME device
    model, criterion = dev_env.to_device(model, nn.CrossEntropyLoss())
    model.to(dev_env.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch_stats_backup = utils.backup_batchnorm_stats(model)
    model.eval()
    return model, optimizer, batch_stats_backup


def get_attack(args, attack_name, dev_env):
    if attack_name:
        eps = args.attack_eps / 255
        lr = args.attack_lr or (1.5 * eps / args.attack_steps)
        attack_criterion = nn.NLLLoss(reduction="sum")
        attack_kwargs = {}
        if attack_name in {"autoattack", "apgd-ce"}:
            attack_kwargs["verbose"] = args.verbose
        else:
            attack_kwargs["dev_env"] = dev_env
        attack = attacks.make_attack(attack_name, eps, lr, args.attack_steps, args.attack_norm, args.attack_boundaries, attack_criterion, **attack_kwargs)
    else:
        attack = None
    return attack

def validate_memo(args, dev_env=None, dataset=None, model=None, loader=None):
    validate_memo_start_time= time.time()
    # might as well try to validate something
    random_seed(args.seed, 0)  # Set all random seeds the same for model/state init (mandatory for XLA)


    args.pretrained = args.pretrained or not args.checkpoint

    dev_env = dev_env or initialize_device(force_cpu=args.force_cpu, amp=args.amp)

    if model is None:
        model = create_model(args.model,
                             pretrained=args.pretrained,
                             num_classes=args.num_classes,
                             in_chans=3,
                             global_pool=args.gp,
                             scriptable=args.torchscript)
        passed_model_none = True
    else:
        passed_model_none = False

    if args.num_classes is None:
        assert hasattr(model,'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint.startswith('gs://') and passed_model_none:
        model = utils.load_model_from_gcs(args.checkpoint,
                                          args.model,
                                          pretrained=args.pretrained,
                                          num_classes=args.num_classes,
                                          in_chans=3,
                                          global_pool=args.gp,
                                          scriptable=args.torchscript)

    elif args.checkpoint and passed_model_none:
        load_checkpoint(model, args.checkpoint, args.use_ema)
    else:
        assert model is not None



    if args.patch_size is not None and isinstance(model, xcit.XCiT) and model.patch_embed.patch_size != args.patch_size:
        assert args.patch_size in {2, 4, 8}, "Finetuning patch size can be only 4, 8 or `None`"
        assert isinstance(model, models.xcit.XCiT), "Finetuning patch size is only supported for XCiT"
        model.patch_embed.patch_size = args.patch_size
        model.patch_embed.proj[0][0].stride = (1, 1)
        if args.patch_size == 4:
            model.patch_embed.proj[2][0].stride = (1, 1)
        if args.patch_size == 2:
            model.patch_embed.proj[4][0].stride = (1, 1)

    param_count = sum([m.numel() for m in model.parameters()])

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)
    data_config['normalize'] = not (args.no_normalize or args.normalize_model)

    if args.normalize_model:
        mean = args.mean or data_config["mean"]
        std = args.std or data_config["std"]
        model = utils.normalize_model(model, mean=mean, std=std)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    # FIXME device
    model, criterion = dev_env.to_device(model, nn.CrossEntropyLoss())
    model.to(dev_env.device)

    dataset = dataset or create_dataset(root=args.data,
                                        name=args.dataset,
                                        split=args.split,
                                        download=args.dataset_download,
                                        load_bytes=args.tf_preprocessing,
                                        class_map=args.class_map)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None
    eval_pp_cfg = utils.MyPreprocessCfg(  # type: ignore
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        crop_pct=1.0 if test_time_pool else data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
        normalize=data_config['normalize'],
    )

    loader = loader or create_loader_v2(dataset,
                                        batch_size=args.batch_size,
                                        is_training=False,
                                        pp_cfg=eval_pp_cfg,
                                        num_workers=args.workers,
                                        pin_memory=args.pin_mem)

    # Not needed for now
    if args.use_mp_loader and dev_env.type_xla:
        import torch_xla.distributed.parallel_loader as pl
        assert isinstance(loader, fetcher.Fetcher)
        loader.use_mp_loader = True
        loader._loader = pl.MpDeviceLoader(loader._loader, dev_env.device)

    if not eval_pp_cfg.normalize:
        loader.dataset.transform.transforms[-1] = transforms.ToTensor()

    tracker = Tracker()
    if dev_env.type_xla:
        from src.metrics_xla import AvgTensorXLA
        losses = AvgTensorXLA(dev_env=dev_env)
        adv_losses = AvgTensorXLA(dev_env=dev_env)
    else:
        losses = AvgTensor()
        adv_losses = AvgTensor()

    accuracy_before = AccuracyTopK(dev_env=dev_env)
    pgd_adv_accuracy_before = AccuracyTopK(dev_env=dev_env)
    AA_adv_accuracy_before = AccuracyTopK(dev_env=dev_env)
    accuracy_after = AccuracyTopK(dev_env=dev_env)
    pgd_adv_accuracy_after = AccuracyTopK(dev_env=dev_env)
    AA_adv_accuracy_after = AccuracyTopK(dev_env=dev_env)


    pgd_attack = get_attack(args=args, attack_name="pgd", dev_env=dev_env)
    AA_attack  = get_attack(args=args, attack_name="autoattack", dev_env=dev_env)
    # Backup batchnorm stats
    batch_stats_backup = utils.backup_batchnorm_stats(model)
    model.eval()
    num_steps = args.num_examples #len(loader)
    if args.num_examples is not None:
        num_steps = min(num_steps, args.num_examples // args.batch_size)
        print(f"Total steps: {num_steps}")

    tracker.mark_iter()
    aug_func = cifar10_aug if "CIFAR10" in args.dataset else imagenet_aug
    te_transforms = transforms.ToTensor() #This is the only needed transform to convert PIL image to tesnor as the dataset is already transformed
    all_l1_norms_before = []
    all_l2_norms_before = []
    all_linf_norms_before = []
    all_l1_norms_after = []
    all_l2_norms_after = []
    all_linf_norms_after = []
    dataset_name = "CIFAR10" if "CIFAR10" in args.dataset else "ImageNet"
    normalize_model_flag = "_normalize_model" if args.normalize_model else ""
    if args.seed>=1:
        results_file_path = f"memo_final_output/vssm_tiny_v2/{dataset_name}/memo_{args.pretraining_type}_pretrained/AdamW_optimizer_{args.reg_type}_reg_num_augs_{args.num_augs}_num_samples_{args.num_examples}{normalize_model_flag}_summary_seed_{args.seed}.txt"
        os.makedirs(os.path.dirname(f'memo_final_output/vssm_tiny_v2/{dataset_name}/memo_{args.pretraining_type}_pretrained'),exist_ok=True)
    else:
        results_file_path = f"memo_output/vssm_tiny_v2/{dataset_name}/memo_{args.pretraining_type}_pretrained/AdamW_optimizer_{args.reg_type}_reg_num_augs_{args.num_augs}_num_samples_{args.num_examples}{normalize_model_flag}_summary.txt"
        os.makedirs(os.path.dirname(f'memo_output/vssm_tiny_v2/{dataset_name}/memo_{args.pretraining_type}_pretrained'),exist_ok=True)
    if not os.path.exists(results_file_path):
        # Create the file if it doesn't exist
        with open(results_file_path, 'w') as file:
            pass  # Just create the file without writing anything
    if args.seed>=1:
        start_index = args.num_examples
    else:
        start_index = 0
    for i in range(start_index,start_index+args.num_examples): # for step_idx, (sample, target) in enumerate(loader):
        model, optimizer, batch_stats_backup = get_initial_model_and_optimizer(args=args, model=None, dev_env=dev_env)
        tracker.mark_iter_data_end()
        image, target = dataset[i]
        image = transforms.ToPILImage()(image) # Convert Tensor/Ndarray to PILImage
        jacobian_l1_norm_before, jacobian_l2_norm_before, jacobian_linf_norm_before = single_eval_test_jacobian_norm(args=args, model=model, device=dev_env.device, image=image, target=target, jacobian_type="gp_correct_max",te_transforms=te_transforms)
        all_l1_norms_before.append(jacobian_l1_norm_before)
        all_l2_norms_before.append(jacobian_l2_norm_before)
        all_linf_norms_before.append(jacobian_linf_norm_before)
        calc_clean_and_adv_accuracy(model=model, sample=image, target=target, te_transforms=te_transforms, accuracy=accuracy_before, pgd_adv_accuracy=pgd_adv_accuracy_before, AA_adv_accuracy=AA_adv_accuracy_before, real_labels=real_labels, valid_labels=valid_labels, pgd_attack=pgd_attack, AA_attack=AA_attack, dev_env=dev_env, batch_stats_backup=batch_stats_backup)
        memo_start_time = time.time()
        adapt_single(args=args, model=model, aug_func=aug_func, optimizer=optimizer, image=image, label=target, cfg=eval_pp_cfg)
        calc_clean_and_adv_accuracy(model=model, sample=image, target=target, te_transforms=te_transforms, accuracy=accuracy_after, pgd_adv_accuracy=pgd_adv_accuracy_after, AA_adv_accuracy=AA_adv_accuracy_after, real_labels=real_labels, valid_labels=valid_labels, pgd_attack=pgd_attack, AA_attack=AA_attack, dev_env=dev_env, batch_stats_backup=batch_stats_backup)
        jacobian_l1_norm_after, jacobian_l2_norm_after, jacobian_linf_norm_after = single_eval_test_jacobian_norm(args=args, model=model, device=dev_env.device, image=image, target=target, jacobian_type="gp_correct_max",te_transforms=te_transforms)
        all_l1_norms_after.append(jacobian_l1_norm_after)
        all_l2_norms_after.append(jacobian_l2_norm_after)
        all_linf_norms_after.append(jacobian_linf_norm_after)

        tracker.mark_iter_step_end()
        tracker.mark_iter()

        # if last_step:
        #     break

    if real_labels is not None:
        top1_before = top5_before = pgd_top1_before = pgd_top5_before = AA_top1_before = AA_top5_before = top1_after = top5_after = pgd_top1_after = pgd_top5_after = AA_top1_after = AA_top5_after = None
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1_before, top5_before = accuracy_before.compute().values()
        pgd_top1_before, pgd_top5_before = pgd_adv_accuracy_before.compute().values()
        AA_top1_before, AA_top5_before = AA_adv_accuracy_before.compute().values()
        top1_after, top5_after = accuracy_after.compute().values()
        pgd_top1_after, pgd_top5_after = pgd_adv_accuracy_after.compute().values()
        AA_top1_after, AA_top5_after = AA_adv_accuracy_after.compute().values()
        print(f"model={args.model} accuracy info:")
        print(f"top1_before={top1_before} pgd_top1_before={pgd_top1_before} AA_top1_before={AA_top1_before}")
        print(f"top1_after={top1_after} pgd_top1_after={pgd_top1_after} AA_top1_after={AA_top1_after}")

    all_l1_norms_before = torch.cat(all_l1_norms_before)
    all_l2_norms_before = torch.cat(all_l2_norms_before)
    all_linf_norms_before = torch.cat(all_linf_norms_before)
    l1_mean_before, l1_std_before = all_l1_norms_before.mean().item(), all_l1_norms_before.std().item()
    l2_mean_before, l2_std_before = all_l2_norms_before.mean().item(), all_l2_norms_before.std().item()
    linf_mean_before, linf_std_before = all_linf_norms_before.mean().item(), all_linf_norms_before.std().item()

    all_l1_norms_after = torch.cat(all_l1_norms_after)
    all_l2_norms_after = torch.cat(all_l2_norms_after)
    all_linf_norms_after = torch.cat(all_linf_norms_after)
    l1_mean_after, l1_std_after = all_l1_norms_after.mean().item(), all_l1_norms_after.std().item()
    l2_mean_after, l2_std_after = all_l2_norms_after.mean().item(), all_l2_norms_after.std().item()
    linf_mean_after, linf_std_after = all_linf_norms_after.mean().item(), all_linf_norms_after.std().item()
    print(f"l1_mean_before={l1_mean_before} l1_std_before={l1_std_before} l1_mean_after={l1_mean_after} l1_std_after={l1_std_after}")
    print(f"l2_mean_before={l2_mean_before} l2_std_before={l2_std_before} l2_mean_after={l2_mean_after} l2_std_after={l2_std_after}")
    print(f"linf_mean_before={linf_mean_before} linf_std_before={linf_std_before} linf_mean_after={linf_mean_after} linf_std_after={linf_std_after}")
    with open(results_file_path, 'a') as file:
        file.write(f"normalize_model={args.normalize_model} pretraining_type={args.pretraining_type} reg_type={args.reg_type} alpha={args.alpha} weight_decay={args.weight_decay} lr={args.lr} niter={args.niter}"
            + f" clean_top1_before={top1_before:.3f} pgd_top1_before={pgd_top1_before:.3f} AA_top1_before={AA_top1_before:.3f} clean_top1_after={top1_after:.3f} pgd_top1_after={pgd_top1_after:.3f} AA_top1_after={AA_top1_after:.3f}"
            + f" l1_mean_before={l1_mean_before} l1_std_before={l1_std_before} l1_mean_after={l1_mean_after} l1_std_after={l1_std_after}"
            + f" l2_mean_before={l2_mean_before} l2_std_before={l2_std_before} l2_mean_after={l2_mean_after} l2_std_after={l2_std_after}"
            + f" linf_mean_before={linf_mean_before} linf_std_before={linf_std_before} linf_mean_after={linf_mean_after} linf_std_after={linf_std_after}" + '\n')

    results = OrderedDict(top1_before=round(top1_before.item(), 4),
                          top1_after=round(top1_after.item(), 4),
                          top5_before=round(top5_before.item(), 4),
                          top5_after=round(top5_after.item(), 4),
                          pgd_top1_before=round(pgd_top1_before.item(), 4),
                          pgd_top1_after=round(pgd_top1_after.item(), 4),
                          pgd_top5_before=round(pgd_top5_before.item(), 4),
                          pgd_top5_after=round(pgd_top5_after.item(), 4),
                          AA_top1_before=round(AA_top1_before.item(), 4),
                          AA_top1_after=round(AA_top1_after.item(), 4),
                          AA_top5_before=round(AA_top5_before.item(), 4),
                          AA_top5_after=round(AA_top5_after.item(), 4),
                          param_count=round(param_count / 1e6, 2),
                          img_size=data_config['input_size'][-1],
                          cropt_pct=eval_pp_cfg.crop_pct,
                          interpolation=data_config['interpolation'])
    print(f"reg_type={args.reg_type} dataset={args.dataset} Program running time: {time.time()-validate_memo_start_time}")
    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    print(args)
    model_cfgs = []
    print("args:")
    print(args)
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= 1:
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate_memo(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt:
            pass
        results = sorted(results, key=lambda x: x['robust_top1_after'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        results = validate_memo(args)
        if args.log_wandb:
            log_results_to_wandb(args, results)


def log_results_to_wandb(args: argparse.Namespace, results: Dict):
    import wandb

    # Get args file from bucket
    assert args.checkpoint.startswith('gs://')
    experiment_dir = os.path.dirname(args.checkpoint)
    args_path = os.path.join(experiment_dir, 'args.yaml')
    with tf.io.gfile.GFile(args_path, mode='r') as f:
        config = yaml.safe_load(f)
    wandb_run_url = config["wandb_run"]
    # Get run identifying info
    if wandb_run_url.endswith('/'):
        wandb_run_url = wandb_run_url[:-1]
    wandb_run_project = wandb_run_url.split("/")[4]
    wandb_run_entity = wandb_run_url.split("/")[3]
    wandb_run_id = wandb_run_url.split("/")[6]
    run = wandb.init(project=wandb_run_project, id=wandb_run_id, entity=wandb_run_entity, resume=True)
    # Log data
    attack = args.attack
    eps = args.attack_eps
    steps = args.attack_steps
    norm = args.attack_norm
    prefix = f"{attack}-{steps}-{eps}"
    if norm != "linf":
        prefix += f"-{norm}"
    dict_to_log = {
        "eval_top1-final": results['top1'],
        f"{prefix}-eval_robust_top1_final": results['robust_top1'],
    }
    run.log(dict_to_log)
    run.finish()


def write_results(results_file, results):
    if results_file.startswith("gs://"):
        open_f = tf.io.gfile.GFile
    else:
        open_f = open
    with open_f(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=list(results[0].keys()))
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


def _mp_entry(*args):
    main()


if __name__ == '__main__':
    program_start_time = time.time()
    main()
    print(f"Program total time: {time.time()-program_start_time}")
