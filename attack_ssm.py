import argparse
import pandas as pd
import numpy as np
import torch
import random
import os
from torchvision import transforms, datasets
import re
from utils.mnist import build_model as  mnist_build_model
from utils.cifar10 import build_model as cifar_build_model
from utils.tiny_imagenet import build_model as tinyimagenet_build_model
from utils.tiny_imagenet import load_tinyimagenet
from utils.evalution import eval_test, adv_test, eval_train, eval_test_jacobian_norm
from pathlib import Path
from autoattack import AutoAttack
import torch.nn as nn

def get_df(excel_file_path, cols_titles):
  if not os.path.isfile(excel_file_path):
    print(f"get_df: excel_file_path is not exist. creating excel_file in {excel_file_path}")
    df = pd.DataFrame(columns=cols_titles)
    df.to_excel(excel_file_path, index=False)
  else:
    df = pd.read_excel(excel_file_path)
  return df


def entry_exists_in_df(df, entry_as_dict):
  exist = True
  for key, value in entry_as_dict.items():
   if isinstance(value[0], float):
     round_value = round(value[0],4)
     df[key] = df[key].fillna(-2) # replacing Nan values with '-2' value
     round_df = df[key].round(4)
     exist = exist and (round_value in round_df.values)
   else:
     exist = exist and (value[0] in df[key].values)

  return exist


def add_entry_to_df(df, entry_as_dict):
  if not entry_exists_in_df(df, entry_as_dict):
    df = pd.concat([df, pd.DataFrame(entry_as_dict)], ignore_index=True)
  else:
    print(f"@@@@@@@@@@@@@@@@@@@@ add_entry_to_df: didn't add new entry to df @@@@@@@@@@@@@@@@@@@@")
  return df


def create_df_entry(cols_titles, entry_values):
  entry_as_dict = {col_title : [entry_values[col]] for col, col_title in enumerate(cols_titles)}
  return entry_as_dict


def get_loader(args):
    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([transforms.ToTensor(),])
        trainset = datasets.CIFAR10(root='datasets/CIFAR10', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        testset = datasets.CIFAR10(root='datasets/CIFAR10', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    elif args.dataset == 'MNIST':
        kwargs = {'num_workers': 1, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.ToTensor()), batch_size=args.test_batch_size,shuffle=False, **kwargs)
    else: #TinyInageNet
        train_loader, test_loader, train_dataset, test_dataset,  num_classes = load_tinyimagenet(args)

    return train_loader, test_loader



def AA_adv_test(model,args,log_path,test_loader,device):
    model.eval()
    correct = 0
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=log_path, version=args.version,verbose=args.AA_verbose,device=device)
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex], bs=args.AA_bs, state_path=args.state_path)
    num_samples = adv_complete.size(0)
    for i in range(0, num_samples, args.AA_bs):
        with torch.no_grad():
            adv_batch = adv_complete[i:i+args.AA_bs].to(device)
            labels_batch = y_test[i:i+args.AA_bs].to(device)
            batch_output = model(adv_batch)
            batch_pred = batch_output.max(1, keepdim=True)[1]
            correct += batch_pred.eq(labels_batch.view_as(batch_pred)).sum().item()
            torch.cuda.empty_cache()

    test_accuracy = correct / args.n_ex
    return test_accuracy

def exploring_adversarial_robustness_evaluation(args, model, device, train_loader, test_loader, log_path, return_train_adv_info=False):
    test_loss, test_acc = eval_test(args, model, device, test_loader)
    print(f"################### Applying AA attack on test_set ###################")
    AA_adv_test_acc = AA_adv_test(model=model, args=args, log_path=log_path, test_loader=test_loader, device=device)
    print(f"################### Applying PGD attack on test_set ###################")
    PGD_adv_test_loss, PGD_adv_test_acc = adv_test(args=args, model=model, device=device, test_loader=test_loader, attack_method="PGD")
    print('================================================================')
    print(f"test_acc={test_acc} PGD_adv_test_acc={PGD_adv_test_acc} AA_adv_test_acc={AA_adv_test_acc}")
    with open(log_path, 'a') as f:
        f.write('Epoch {}, Test Acc: {:.4f}, PGD Adv Test Acc: {:.4f}, AA Adv Test Acc: {:.4f}\n'.format(args.epochs, test_acc, PGD_adv_test_acc, AA_adv_test_acc))
    if return_train_adv_info:
        train_loss, train_acc = eval_train(args, model, device, train_loader)
    else:
        train_loss = train_acc = -1

    return train_loss, train_acc, 100. * test_acc, 100. * PGD_adv_test_acc, 100. * AA_adv_test_acc


def extract_dir_info(dir):
    dir_splited = dir.split('_')
    lr = dir_splited[1]
    alpha = dir_splited[3]
    if "auxiliary" in dir:
        aux_beta = dir_splited[-1]
    else:
        aux_beta = -1
    return alpha, aux_beta, lr


def get_model(args, ckpt_path, device):
    if args.dataset == "MNIST":
        model = mnist_build_model(args, args.model_name).to(device)
    elif args.dataset == "CIFAR10":
        model = cifar_build_model(args, args.model_name).to(device)
    elif args.dataset == "TinyImageNet":
        model = tinyimagenet_build_model(args, args.model_name).to(device)
    else:
        model = None
        print(f"get_model: unknown dataset: {args.dataset} !")
        exit(1)
    state_dict = torch.load(ckpt_path)
    if 'module.' in list(state_dict.keys())[0]:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    return model.eval().to(device)


def transfer_attack_multiple_seeds(args):
    if args.finetuning:
        if args.constant_lr:
            finetuning_prefix = f"finetuning_constant_lr_"
        else:
            finetuning_prefix = f"finetuning_"
    else:
        finetuning_prefix = ""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    generator = torch.Generator()
    generator.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_seeds = range(5,5+args.num_of_seeds)
    log_path = f"checkpoints/{args.model_name}/{args.dataset}/{finetuning_prefix}{args.epochs}_epochs/{args.reg_type}/{args.AT_type}/{args.AT_type}_reg_{args.reg_type}_{finetuning_prefix}epochs_{args.epochs}_lr_{args.lr}_thresh_{args.thresh}_attacks_summary_log_{args.gpu_type}_{args.n_ex}_samples_AA_bs{args.AA_bs}.txt"
    excel_file_path = f"checkpoints/{args.model_name}/{args.dataset}/{finetuning_prefix}{args.epochs}_epochs/{args.reg_type}/{args.AT_type}/{args.AT_type}_reg_{args.reg_type}_{finetuning_prefix}epochs_{args.epochs}_lr_{args.lr}_thresh_{args.thresh}_attacks_summary_{args.gpu_type}_{args.n_ex}_samples_AA_bs{args.AA_bs}.xlsx"
    cols_titles = ["architecture", "model_seeds", "train_loss_mean", "train_loss_std", "train_acc_mean", "train_acc_std", "val_acc_mean", "val_acc_std", "PGD_mean", "PGD_std", "avg_AutoAttack", "std_AutoAttack", "jacobian_l1_mean", "jacobian_l1_std", "jacobian_l2_mean", "jacobian_l2_std", "jacobian_linf_mean", "jacobian_linf_std", "lr", "alpha", "aux_beta", "thresh", "epochs" , "train_gpu", "attacks_gpu"]
    results_df = get_df(excel_file_path, cols_titles)
    train_loader, test_loader = get_loader(args)
    train_gpu = "unknown"
    for dir in os.listdir(f"checkpoints/{args.model_name}/{args.dataset}/{finetuning_prefix}{args.epochs}_epochs/{args.reg_type}/{args.AT_type}"): # going over the different experiments
        if dir.endswith(".xlsx") or dir.endswith(".txt"):
            continue

        alpha, aux_beta, lr = extract_dir_info(dir)
        if float(lr) != float(args.lr):
            print(f"target_lr={args.lr} current_lr={lr} --> continue")
            continue
        acc_results = {'Clean': [], 'PGD': [], 'AA': []}
        train_info = {'train_loss': [], 'train_acc': []}
        jacobian_norms_info = {'jacobian_l1': [], 'jacobian_l2': [], 'jacobian_linf': []}
        for file_name in os.listdir(f"checkpoints/{args.model_name}/{args.dataset}/{finetuning_prefix}{args.epochs}_epochs/{args.reg_type}/{args.AT_type}/{dir}"): # going over the different weights seeds
            current_thresh = re.findall(r'_thresh_([\d]+)', file_name)
            current_seed = re.findall(r'seed_([\d.]+)', file_name)
            epochs = re.findall(r'-epoch(\d+)', file_name)
            print(f"current_thresh={current_thresh} current_seed={current_seed} epochs={epochs}")
            if not file_name.endswith(".pt"):
                continue
            if args.thresh==0 and len(current_thresh)>0:
                print(f"continue due to thresh bigger than 0")
                continue
            if args.thresh>0 and not (len(current_thresh)>0 and int(current_thresh[0]) == args.thresh):
                print(f"continue due to wrong thresh")
                continue

            if len(epochs) > 0 and int(epochs[0]) != args.epochs:
                print(f"continue due to wrong epochs")
                continue
            train_gpu_extraction = re.findall(r"gpu(.*?)\.pt", file_name)
            if train_gpu == "unknown" and len(train_gpu_extraction) > 0:
                train_gpu = train_gpu_extraction[0]
            ckpt_path = f"checkpoints/{args.model_name}/{args.dataset}/{finetuning_prefix}{args.epochs}_epochs/{args.reg_type}/{args.AT_type}/{dir}/{file_name}"
            print(f"ckpt_path={ckpt_path}")
            # Load the model for each seed
            model = get_model(args, ckpt_path, device)
            train_loss, train_acc, clean_test_acc, PGD_adv_test_acc, AA_adv_test_acc = exploring_adversarial_robustness_evaluation(args=args, model=model, device=device, train_loader=train_loader, test_loader=test_loader, log_path=log_path, return_train_adv_info=False)
            _, (jacobian_l1_mean, jacobian_l1_std), (jacobian_l2_mean, jacobian_l2_std), (
                jacobian_linf_mean, jacobian_linf_std) = eval_test_jacobian_norm(args, model, device, test_loader,
                                                                                 args.reg_type)
            jacobian_norm_stats = {"jacobian_l1_mean": jacobian_l1_mean,
                                    "jacobian_l1_std": jacobian_l1_std,
                                    "jacobian_l2_mean": jacobian_l2_mean,
                                    "jacobian_l2_std": jacobian_l2_std,
                                    "jacobian_linf_mean": jacobian_linf_mean,
                                    "jacobian_linf_std": jacobian_linf_std}
            # Set the attack parameters
            print(f"file_name={file_name}  clean_test_acc={clean_test_acc} PGD_adv_test_acc={PGD_adv_test_acc} AA_adv_test_acc={AA_adv_test_acc}")
            acc_results["Clean"].append(clean_test_acc)
            acc_results["PGD"].append(PGD_adv_test_acc)
            acc_results["AA"].append(AA_adv_test_acc)
            train_info["train_loss"].append(train_loss)
            train_info["train_acc"].append(train_acc)
            jacobian_norms_info["jacobian_l1"].append(jacobian_norm_stats["jacobian_l1_mean"])
            jacobian_norms_info["jacobian_l2"].append(jacobian_norm_stats["jacobian_l2_mean"])
            jacobian_norms_info["jacobian_linf"].append(jacobian_norm_stats["jacobian_linf_mean"])

        # Calculate mean and std for the current attack and model
        train_loss_mean = np.mean(train_info["train_loss"])
        train_acc_mean = np.mean(train_info["train_acc"])
        clean_mean = np.mean(acc_results["Clean"])
        PGD_mean = np.mean(acc_results["PGD"])
        AA_mean = np.mean(acc_results["AA"])
        jacobian_l1_mean = np.mean(jacobian_norms_info["jacobian_l1"])
        jacobian_l2_mean = np.mean(jacobian_norms_info["jacobian_l2"])
        jacobian_linf_mean = np.mean(jacobian_norms_info["jacobian_linf"])
        if len(acc_results["Clean"]) > 1:
            train_loss_std = np.std(train_info["train_loss"])
            train_acc_std = np.std(train_info["train_acc"])
            clean_std = np.std(acc_results["Clean"])
            PGD_std = np.std(acc_results["PGD"])
            AA_std = np.std(acc_results["AA"])
            jacobian_l1_std = np.std(jacobian_norms_info["jacobian_l1"])
            jacobian_l2_std = np.std(jacobian_norms_info["jacobian_l2"])
            jacobian_linf_std = np.std(jacobian_norms_info["jacobian_linf"])
        else:
            train_loss_std = train_acc_std = clean_std = PGD_std = AA_std = jacobian_l1_std = jacobian_l2_std = jacobian_linf_std =  -1

        df_values = [f"{args.model_name}_{dir}_{args.reg_type}", ",".join(map(str, target_seeds)), round(train_loss_mean,3), round(train_loss_std,3), round(train_acc_mean,3), round(train_acc_std,3), round(clean_mean,3), round(clean_std,3), round(PGD_mean,3), round(PGD_std,3), round(AA_mean,3), round(AA_std,3), round(jacobian_l1_mean,3), round(jacobian_l1_std,3), round(jacobian_l2_mean,3), round(jacobian_l2_std,3), round(jacobian_linf_mean,3), round(jacobian_linf_std,3),  float(lr), float(alpha), float(aux_beta), args.thresh, f"{finetuning_prefix}{args.epochs}", train_gpu, args.gpu_type]
        df_entry_as_dict = create_df_entry(cols_titles, df_values)
        print(f"###### adding entry to df ######")
        results_df = add_entry_to_df(results_df, df_entry_as_dict)

    results_df.to_excel(excel_file_path, index=False)


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')
    parser.add_argument('--constant_lr', action='store_true', default=False)
    parser.add_argument('--finetuning', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu_type', type=str, default="a_100")
    parser.add_argument('--num_of_seeds', type=int, default=1)
    parser.add_argument('--thresh', type=float, default=0)
    parser.add_argument('--reg_type', default='none', type=str)
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=256, metavar='N')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N')
    parser.add_argument('--epochs', type=int, default=180, metavar='N', help='number of epochs to train')
    parser.add_argument('--workers', type=int, default=12, metavar='N', help='number of epochs to train')
    parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--epsilon', default=8 / 255, type=float, help='perturbation')
    parser.add_argument('--num_steps', default=10, type=int, help='perturb number of steps')
    parser.add_argument('--step_size', default=2 / 255, type=float, help='perturb step size')
    parser.add_argument('--beta', default=6.0, type=int, help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--seed', type=int, default=5, metavar='S', help='random seed (default: 5)')
    parser.add_argument('--num_layers', type=int, default=4, help='layers of SSM')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--model-dir', default='checkpoints', help='directory of model for saving checkpoint')
    parser.add_argument('--AT_type', default='Nat', type=str, help='Madry, TRADE, FreeAT or YOPO in adversrial training stage')
    parser.add_argument('--attack_type', default='PGD', type=str, help='PGD or AA attack in test stage')
    parser.add_argument('--model_name', default='S6', type=str, help='SSM, DSS, S5, Mega, S6')
    parser.add_argument('--AA_lags', type=int, default=4, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--use_AdSS', action='store_true', default=False, help='use AdSS module')
    parser.add_argument('--AdSS_Type', type=str, choices=['relu', 'sgmd', 'tanh'], default='relu', help='the AdSS type used in training')
    # ! AA evaluation part
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--n_ex', type=int, default=2000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--AA_bs', type=int, default=200)
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)
    parser.add_argument('--AA_verbose', action='store_true', default=True, help='use the verbose in AA progress')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    print(f"attack_ssm.py args:")
    print(args)
    transfer_attack_multiple_seeds(args)
    print(f"Finished!")
