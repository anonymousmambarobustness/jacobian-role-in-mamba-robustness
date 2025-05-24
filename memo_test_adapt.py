from __future__ import print_function
from torchvision import transforms
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from PIL import Image
from tqdm import tqdm

from utils.memo_prepare_dataset import cifar10_prepare_test_data, get_cifar10_te_transforms, tinyimagenet_prepare_test_data, get_tinyimagenet_te_transforms
from utils.memo_third_party import cifar10_aug, single_eval_test_jacobian_norm
from utils.memo_tinyimagenet_third_party import tinyimagenet_aug
from utils.trades import regularization_term
from utils.used_attacks import PGD
from utils.mnist import build_model as  mnist_build_model
from utils.cifar10 import build_model as cifar_build_model
from utils.tiny_imagenet import build_model as tiny_imagenet_build_model
from pathlib import Path
from autoattack import AutoAttack
import time
def get_ssm_model(args, ckpt_path, device):
    if args.dataset == "MNIST":
        model = mnist_build_model(args, args.model_name).to(device)
    elif args.dataset == "CIFAR10":
        model = cifar_build_model(args, args.model_name).to(device)
    elif args.dataset == "TinyImageNet":
        model = tiny_imagenet_build_model(args, args.model_name).to(device)
    else:
        model = None
        print(f"get_ssm_model: unknown dataset: {args.dataset} !")
        exit(1)
    state_dict = torch.load(ckpt_path)
    if 'module.' in list(state_dict.keys())[0]:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    return model.eval().to(device)

parser = argparse.ArgumentParser()
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--dataroot', required=True)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--group_norm', default=8, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--niter', default=1, type=int)

parser.add_argument('--num_samples', type=int, default=100, help='layers of SSM')
parser.add_argument('--num_layers', type=int, default=4, help='layers of SSM')
parser.add_argument('--reg_type', default='none', type=str)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--model_name', default='S6', type=str, help='SSM, DSS, S5, Mega, S6')
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--normalize_data', action='store_true', default=False)
parser.add_argument('--use_AdSS', action='store_true', default=False, help='use AdSS module')
parser.add_argument('--AdSS_Type', type=str, choices=['relu', 'sgmd', 'tanh'], default='relu', help='the AdSS type used in training')
parser.add_argument('--epsilon', default=8 / 255, type=float, help='perturbation')
parser.add_argument('--num_steps', default=10, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=2 / 255, type=float, help='perturb step size')
parser.add_argument('--weight_decay', default=2e-4, type=float)
parser.add_argument('--optimizer', default="AdamW", type=str)
parser.add_argument('--AT_type', default='Nat', type=str, help='adversrial training stage')
parser.add_argument('--workers', type=int, default=1, metavar='N')
parser.add_argument('--num_classes', type=int, default=10, metavar='N')
parser.add_argument('--seed', type=int, default=0, metavar='N')


# AA evaluation
parser.add_argument('--calc_AA', action='store_true', default=False)
parser.add_argument('--norm', type=str, default='Linf')
parser.add_argument('--version', type=str, default='standard')
parser.add_argument('--state-path', type=Path, default=None)
parser.add_argument('--AA_verbose', action='store_true', default=True, help='use the verbose in AA progress')

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
generator = torch.Generator()
generator.manual_seed(args.seed)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.dataset == "CIFAR10":
    te_transforms = get_cifar10_te_transforms(args)
    teset, teloader = cifar10_prepare_test_data(args=args, te_transforms=te_transforms)
    ckpt_path = f"checkpoints/{args.model_name}/CIFAR10/180_epochs/none/{args.AT_type}/lr_0.001_alpha_0/{args.model_name}{args.AT_type}-epoch180.pt"
    aug = cifar10_aug
elif args.dataset == "TinyImageNet":
    te_transforms = get_tinyimagenet_te_transforms(args)
    teset, teloader = tinyimagenet_prepare_test_data(args)
    ckpt_path = f"checkpoints/{args.model_name}/TinyImageNet/180_epochs/none/{args.AT_type}/lr_0.001_alpha_0/{args.model_name}{args.AT_type}-epoch180.pt"
    aug = tinyimagenet_aug
else:
    te_transforms = None
    teset, teloader = None, None
    ckpt_path = None
    aug = None
    print(f"Unknown dataset: {args.dataset} -> exit program!")
    exit(1)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def adapt_single(args,aug,model,optimizer,image,label):
    model.eval()
    for iteration in range(args.niter):
        inputs = [aug(image,args) for _ in range(args.batch_size)]
        inputs = torch.stack(inputs).cuda()
        optimizer.zero_grad()
        if args.reg_type in ["gp","gp_correct_class", "gp_correct_class_l1", "gp_correct_class_l2", "gp_max_output"]:
            inputs.requires_grad = True
        outputs = model(inputs)
        loss, logits = marginal_entropy(outputs)
        if args.reg_type in ["gp", "gp_correct_class", "gp_correct_class_l1", "gp_correct_class_l2", "gp_max_output"]:
            labels_expanded = torch.tensor(label).repeat(args.batch_size)  # Assumes label is a single tensor scalar
            reg_val = regularization_term(inputs=inputs, outputs=outputs, partial_inputs=None, partial_outputs=None, targets=labels_expanded, regularization_type=args.reg_type, model=model, args=args)
            original_loss = loss.item()
            loss = loss + (args.alpha * reg_val)
            inputs.requires_grad = False
            print(f"iteration{iteration}: loss={loss} original_loss={original_loss} args.alpha*reg_val={args.alpha * reg_val} reg_val={reg_val}  alpha={args.alpha}")
        else:
            print(f"iteration{iteration}: loss={loss} ")
        loss.backward()
        optimizer.step()


def test_single(args, model, image, label, te_transforms, AA_log_path=None, attack_name="none"):
    model.eval()
    inputs = te_transforms(image).unsqueeze(0).cuda()
    label_expanded = torch.tensor(label).repeat(inputs.shape[0]).to(inputs.device)
    if attack_name == "PGD":
        PGD_attack = PGD(model, args.epsilon, args.step_size, args.num_steps, normalize=args.normalize_data, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #TODO: set normalize=False permanently as in exploring case
        inputs = PGD_attack(inputs, label_expanded)
    elif attack_name == "AA":
        start_time = time.time()
        AA_attack = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=AA_log_path, version=args.version, verbose=args.AA_verbose)
        inputs = AA_attack.run_standard_evaluation(inputs, label_expanded, bs=inputs.shape[0], state_path=args.state_path)
    else:
        pass

    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
    correctness = 1 if predicted.item() == label else 0
    return correctness, confidence



def get_correctness(args, model, image, label, te_transforms, AA_log_path):
    clean_correctness, _ = test_single(args=args, model=model, image=image, label=label, te_transforms=te_transforms)
    PGD_correctness, _ = test_single(args=args, model=model, image=image, label=label, te_transforms=te_transforms, attack_name="PGD")
    if args.calc_AA:
        AA_correctness, _ = test_single(args=args, model=model, image=image, label=label, te_transforms=te_transforms, AA_log_path=AA_log_path, attack_name="AA")
    else:
        AA_correctness = -1
    return clean_correctness, PGD_correctness, AA_correctness





print('Running...')
clean_correct_before_adapt = []
PGD_correct_before_adapt = []
AA_correct_before_adapt = []

clean_correct_after_adapt = []
PGD_correct_after_adapt = []
AA_correct_after_adapt = []


normalize_data_suffix = "_normalized_data" if args.normalize_data else ""
if args.seed >= 1:
    start_index = 500
    results_file_path = f"memo_final_results/{args.model_name}/{args.dataset}/memo_{args.AT_type}_pretrained/{args.optimizer}_optimizer_{args.reg_type}_reg_batch_size_{args.batch_size}_num_samples_{args.num_samples}{normalize_data_suffix}_summary_seed_{args.seed}.txt"
    AA_log_path = f'memo_AA_logs/{args.model_name}/{args.dataset}/{args.AT_type}_pretrained/{args.model_name}_{args.num_samples}_samples_{args.reg_type}_reg_niter_{args.niter}_weight_decay_{args.weight_decay}_lr_{args.lr}_alpha_{args.alpha}_AA_log_seed_{args.seed}.txt'
else:
    start_index = 0
    results_file_path = f"checkpoints/{args.model_name}/{args.dataset}/memo_{args.AT_type}_pretrained/{args.optimizer}_optimizer_{args.reg_type}_reg_batch_size_{args.batch_size}_num_samples_{args.num_samples}{normalize_data_suffix}_summary.txt"
    AA_log_path = f'memo_AA_logs/{args.model_name}/{args.dataset}/{args.AT_type}_pretrained/{args.model_name}_{args.num_samples}_samples_{args.reg_type}_reg_niter_{args.niter}_weight_decay_{args.weight_decay}_lr_{args.lr}_alpha_{args.alpha}_AA_log.txt'
os.makedirs(os.path.dirname(f'memo_AA_logs/{args.model_name}/{args.dataset}/{args.AT_type}_pretrained'), exist_ok=True)
os.makedirs(os.path.dirname(f'checkpoints/{args.model_name}/{args.dataset}/memo_{args.AT_type}_pretrained'), exist_ok=True)
if not os.path.exists(AA_log_path):
    with open(AA_log_path, 'w') as file:
        pass  # Just create the file without writing anything
if not os.path.exists(results_file_path):
    # Create the file if it doesn't exist
    with open(results_file_path, 'w') as file:
        pass  # Just create the file without writing anything
program_start_time = time.time()
all_l1_norms_before = []
all_l2_norms_before = []
all_linf_norms_before = []
all_l1_norms_after = []
all_l2_norms_after = []
all_linf_norms_after = []
class_counts = {i: 0 for i in range(args.num_classes)}
for i in range(start_index, start_index + args.num_samples):
    sample_process_start_time = time.time()

    net = get_ssm_model(args=args, ckpt_path=ckpt_path, device=device)
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr)

    if args.dataset == "CIFAR10":
        _, label = teset[i]
        image = Image.fromarray(teset.data[i])
    elif args.dataset == "TinyImageNet":
        image, label = teset[i]
        image = transforms.ToPILImage()(image) # Convert Tensor/Ndarray to PILImage

    else:
        label = None
        print(f"dataset:{args.dataset} is not supported")
        exit(1)

    print(f"################## finish processing seed={args.seed} start_index={start_index} sample {i} model={args.model_name} nug_augs={args.batch_size} niter={args.niter} proccess_time={time.time()-sample_process_start_time}##################")
    class_counts[label] += 1
    jacobian_l1_norm_before, jacobian_l2_norm_before, jacobian_linf_norm_before = single_eval_test_jacobian_norm(args=args, model=net, device=device, image=image, target=label, jacobian_type="gp_correct_max", te_transforms=te_transforms)
    all_l1_norms_before.append(jacobian_l1_norm_before)
    all_l2_norms_before.append(jacobian_l2_norm_before)
    all_linf_norms_before.append(jacobian_linf_norm_before)
    clean_correctness_before_adapt, PGD_correctness_before_adapt, AA_correctness_before_adapt = get_correctness(args=args, model=net, image=image, label=label, te_transforms=te_transforms, AA_log_path=AA_log_path)
    clean_correct_before_adapt.append(clean_correctness_before_adapt)
    PGD_correct_before_adapt.append(PGD_correctness_before_adapt)
    AA_correct_before_adapt.append(AA_correctness_before_adapt)
    single_sample_memo_adapt_start_time = time.time()
    adapt_single(args=args,aug=aug,model=net,optimizer=optimizer,image=image,label=label)
    get_correctness_time = time.time()
    clean_correctness_after_adapt, PGD_correctness_after_adapt, AA_correctness_after_adapt = get_correctness(args=args, model=net, image=image, label=label, te_transforms=te_transforms, AA_log_path=AA_log_path)
    clean_correct_after_adapt.append(clean_correctness_after_adapt)
    PGD_correct_after_adapt.append(PGD_correctness_after_adapt)
    AA_correct_after_adapt.append(AA_correctness_after_adapt)
    jacobian_l1_norm_after, jacobian_l2_norm_after, jacobian_linf_norm_after = single_eval_test_jacobian_norm(args=args, model=net, device=device, image=image, target=label, jacobian_type="gp_correct_max", te_transforms=te_transforms)
    all_l1_norms_after.append(jacobian_l1_norm_after)
    all_l2_norms_after.append(jacobian_l2_norm_after)
    all_linf_norms_after.append(jacobian_linf_norm_after)

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

print(f"model={args.model_name} pretrained_AT_type={args.AT_type} optimizer={args.optimizer} normalize_data={args.normalize_data} weight_decay={args.weight_decay} reg_type={args.reg_type} alpha={args.alpha} lr={args.lr} niter={args.niter}")
print(f"before_adapt_clean_test_accuracy {(np.mean(clean_correct_before_adapt))*100:.3f} before_adapt_PGD_accuracy={(np.mean(PGD_correct_before_adapt))*100:.3f} before_adapt_AA_accuracy={(np.mean(AA_correct_before_adapt))*100:.3f}")
print(f"after_adapt_clean_test_accuracy {(np.mean(clean_correct_after_adapt))*100:.3f} after_adapt_PGD_accuracy={(np.mean(PGD_correct_after_adapt))*100:.3f} after_adapt_AA_accuracy={(np.mean(AA_correct_after_adapt))*100:.3f}")
print(f"l1_mean_before={l1_mean_before} l1_std_before={l1_std_before} l1_mean_after={l1_mean_after} l1_std_after={l1_std_after}")
print(f"l2_mean_before={l2_mean_before} l2_std_before={l2_std_before} l2_mean_after={l2_mean_after} l2_std_after={l2_std_after}")
print(f"linf_mean_before={linf_mean_before} linf_std_before={linf_std_before} linf_mean_after={linf_mean_after} linf_std_after={linf_std_after}")
with open(results_file_path, 'a') as file:
    file.write(f"model={args.model_name} normalize_data={args.normalize_data} pretrained_AT_type={args.AT_type} reg_type={args.reg_type} alpha={args.alpha} lr={args.lr} weight_decay={args.weight_decay} niter={args.niter} before_adapt_clean_test_accuracy {(np.mean(clean_correct_before_adapt))*100:.2f} before_adapt_PGD_accuracy={(np.mean(PGD_correct_before_adapt))*100:.2f} before_adapt_AA_accuracy={(np.mean(AA_correct_before_adapt))*100:.2f} after_adapt_clean_test_accuracy={(np.mean(clean_correct_after_adapt))*100:.2f} after_adapt_PGD_accuracy={(np.mean(PGD_correct_after_adapt))*100:.2f} after_adapt_AA_accuracy={(np.mean(AA_correct_after_adapt))*100:.2f}"
               + f" l1_mean_before={l1_mean_before} l1_std_before={l1_std_before} l1_mean_after={l1_mean_after} l1_std_after={l1_std_after}"
               + f" l2_mean_before={l2_mean_before} l2_std_before={l2_std_before} l2_mean_after={l2_mean_after} l2_std_after={l2_std_after}"
               + f" linf_mean_before={linf_mean_before} linf_std_before={linf_std_before} linf_mean_after={linf_mean_after} linf_std_after={linf_std_after}" + '\n')

print(f"class counts: {class_counts}")
print(f'Program running time:{time.time()-program_start_time}')
