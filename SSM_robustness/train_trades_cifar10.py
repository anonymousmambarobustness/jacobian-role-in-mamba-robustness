import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms
from autoattack import AutoAttack
from utils.args import Build_Parser
from utils.evalution import eval_test, adv_test, adv_eval_train, eval_train
from utils.cifar10 import build_model
from utils.trades import train, test_time_jacobian_fintune_epoch
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np

# args
parser = Build_Parser()
args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='datasets/CIFAR10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers = 12, pin_memory = True)
testset = torchvision.datasets.CIFAR10(root='datasets/CIFAR10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers = 12, pin_memory = True)

def AA_adv_test(model,args,log_path):
    model.eval()
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=log_path,
        version=args.version,verbose=args.AA_verbose)
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    torch.cuda.empty_cache()
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.AA_bs, state_path=args.state_path)

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)
            

def plot_loss_components(output_dir, epochs, losses_dict, title, filename):
    plt.figure(figsize=(12, 8))
    for label, values in losses_dict.items():
        values = values if isinstance(values, list) else values.tolist()
        plt.plot(epochs, values, marker='o', linestyle='-', label=label)

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(save_path)
    plt.close()


def plot_all_loss_curves(model_dir, experiment_name, epochs, train_losses, train_original_losses, train_normalized_reg_vals, train_reg_vals):
    """
    Plots all combinations of training loss components using plot_loss_components.
    """
    # Plot combinations with normalized reg
    plot_loss_components(
        output_dir=model_dir,
        epochs=epochs,
        losses_dict={
            "Train Loss": train_losses,
            "Normalized Reg Val (Reg × α)": train_normalized_reg_vals
        },
        title="Train Loss vs Normalized Regularization",
        filename=f"{experiment_name}_loss_and_normalized_reg"
    )
    plot_loss_components(
        output_dir=model_dir,
        epochs=epochs,
        losses_dict={
            "Train Original Loss": train_original_losses,
            "Normalized Reg Val (Reg × α)": train_normalized_reg_vals
        },
        title="Original Loss vs Normalized Regularization",
        filename=f"{experiment_name}_orig_loss_and_normalized_reg"
    )
    plot_loss_components(
        output_dir=model_dir,
        epochs=epochs,
        losses_dict={
            "Train Loss": train_losses,
            "Train Original Loss": train_original_losses,
            "Normalized Reg Val (Reg × α)": train_normalized_reg_vals
        },
        title="All Loss Components",
        filename=f"{experiment_name}_all_losses"
    )

    # Plot combinations with unnormalized reg
    plot_loss_components(
        output_dir=model_dir,
        epochs=epochs,
        losses_dict={
            "Train Loss": train_losses,
            "Reg Val": train_reg_vals
        },
        title="Train Loss vs Reg Val",
        filename=f"{experiment_name}_loss_and_reg_val"
    )
    plot_loss_components(
        output_dir=model_dir,
        epochs=epochs,
        losses_dict={
            "Train Original Loss": train_original_losses,
            "Reg Val": train_reg_vals
        },
        title="Original Loss vs Reg Val",
        filename=f"{experiment_name}_orig_loss_and_reg_val"
    )
    plot_loss_components(
        output_dir=model_dir,
        epochs=epochs,
        losses_dict={
            "Train Loss": train_losses,
            "Train Original Loss": train_original_losses,
            "Reg Val": train_reg_vals
        },
        title="All Loss Components (with Reg Val)",
        filename=f"{experiment_name}_all_losses_with_reg_val"
    )

    # Individual plots
    plot_loss_components(
        output_dir=model_dir,
        epochs=epochs,
        losses_dict={"Train Loss": train_losses},
        title="Train Loss",
        filename=f"{experiment_name}_train_loss"
    )
    plot_loss_components(
        output_dir=model_dir,
        epochs=epochs,
        losses_dict={"Train Original Loss": train_original_losses},
        title="Train Original Loss",
        filename=f"{experiment_name}_train_original_loss"
    )
    plot_loss_components(
        output_dir=model_dir,
        epochs=epochs,
        losses_dict={"Normalized Reg Val (Reg × α)": train_normalized_reg_vals},
        title="Normalized Regularization",
        filename=f"{experiment_name}_normalized_reg_val"
    )
    plot_loss_components(
        output_dir=model_dir,
        epochs=epochs,
        losses_dict={"Reg Val": train_reg_vals},
        title="Regularization Term",
        filename=f"{experiment_name}_reg_val"
    )



def plot_loss_curves(output_dir, epochs, train_losses, train_original_losses, train_reg_vals, experiment_name):
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, marker='o', linestyle='-', color='r', label='Train Loss')
    plt.plot(epochs, train_original_losses, marker='o', linestyle='-', color='y', label='Train Original Loss')
    plt.plot(epochs, train_reg_vals, marker='o', linestyle='-', color='g', label='Train reg vals')
    plt.title("Loss")
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_epochs_{len(train_losses)}_loss.png'))
    plt.close()


def plot_accuracy_curves(output_dir, epochs, test_accuracies, experiment_name):
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, test_accuracies, marker='o', linestyle='-', color='r', label='Test Accuracy')
    plt.title("Test Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_epochs_{len(test_accuracies)}_acc.png'))
    plt.close()


def plot_grads_norm_curve(output_dir, epochs, grad_norms, experiment_name):
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, grad_norms, marker='o', linestyle='-', color='r', label='Test Accuracy')
    plt.title("Total Grads Norm")
    plt.xlabel('Epochs')
    plt.ylabel("Grads Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{experiment_name}_epochs_{len(grad_norms)}_grads_norm.png'))
    plt.close()


def eval_gp_correct_class(inputs, outputs, targets):
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


def eval_gp_max_output(inputs, outputs):
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


def eval_test_jacobian_norm(args, model, device, test_loader, jacobian_type):
    model.eval()
    correct = 0
    all_l1_norms = []
    all_l2_norms = []
    all_linf_norms = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        # Compute Jacobian for model
        if "gp_max_output" in jacobian_type:
            jacobian_row = eval_gp_max_output(data, output)
        else:
            jacobian_row = eval_gp_correct_class(data, output, target)
        # Compute L1, L2, and Linf norms
        jacobian_l1_norm = jacobian_row.view(args.batch_size, -1).norm(1, 1) # (p=1,dim=1)
        jacobian_l2_norm = jacobian_row.view(args.batch_size, -1).norm(2, 1)
        jacobian_linf_norm = jacobian_row.view(args.batch_size, -1).abs().max(dim=1)[0]  # L-infinity norm

        # Store all norms
        all_l1_norms.append(jacobian_l1_norm)
        all_l2_norms.append(jacobian_l2_norm)
        all_linf_norms.append(jacobian_linf_norm)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        data.requires_grad = False
        torch.cuda.empty_cache()

    # Concatenate all batch norms to a single tensor
    all_l1_norms = torch.cat(all_l1_norms)
    all_l2_norms = torch.cat(all_l2_norms)
    all_linf_norms = torch.cat(all_linf_norms)
    # Compute mean and std for each norm
    l1_mean, l1_std = all_l1_norms.mean().item(), all_l1_norms.std().item()
    l2_mean, l2_std = all_l2_norms.mean().item(), all_l2_norms.std().item()
    linf_mean, linf_std = all_linf_norms.mean().item(), all_linf_norms.std().item()
    test_accuracy = correct / len(test_loader.dataset)
    return 100*test_accuracy, (l1_mean, l1_std), (l2_mean, l2_std), (linf_mean, linf_std)



def main():
    auxiliary_beta_str = f"_auxiliary_beta_{args.auxiliary_beta}" if args.model_name == "S6_SSM_auxiliary" else ""
    experiment_name = f"{args.model_name}_{args.AT_type}_CIFAR10_reg_{args.reg_type}_alpha_{args.alpha}{auxiliary_beta_str}_lr_{args.lr}_epochs_{args.epochs}_seed_{args.seed}_thresh_{args.thresh}"
    seed_thresh_info = f"seed_{args.seed}_thresh_{int(args.thresh)}" if args.thresh > 0 else f"seed_{args.seed}"
    if args.use_AdSS:
        log_path = os.path.join(model_dir,f"{args.model_name}_reg_{args.reg_type}_alpha_{args.alpha}{auxiliary_beta_str}_lr_{args.lr}_{'AdSS_{}'.format(args.AdSS_Type)}_{args.AT_type}_{seed_thresh_info}_train_log.txt")
    else:
        log_path = os.path.join(model_dir,f"{args.model_name}_reg_{args.reg_type}_alpha_{args.alpha}{auxiliary_beta_str}_lr_{args.lr}_{args.AT_type}_{seed_thresh_info}_train_log.txt")
    model = build_model(args, args.model_name).to(device)
    if args.finetuning:
        original_lr = 1e-3
        split_model_name = args.model_name.split("_")
        original_epochs = 300 if (len(split_model_name)>1 and split_model_name[1] == "1") else 180
        AT_type = "Nat" if args.AT_type == "Nat_test_time_jacobian" else args.AT_type
        ckpt_path = f"checkpoints/{args.model_name}/CIFAR10/{original_epochs}_epochs/none/{AT_type}/lr_{original_lr}_alpha_{0}/{args.model_name}{AT_type}-epoch{original_epochs}.pt"
        state_dict = torch.load(ckpt_path)
        if 'module.' in list(state_dict.keys())[0]:
            model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
        loaded_model_test_loss, loaded_model_test_acc = eval_test(args, model, device, test_loader)
        print(f"Finetuning mode: loaded_model_test_acc={(100. * loaded_model_test_acc)}")
        if (100. * loaded_model_test_acc) < 40:
            print(f"loaded_model_test_acc is smaller than 40!!! -> There is an error during load stage -> exit program")
            exit(1)



    elif torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
            
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.finetuning and args.constant_lr:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    train_losses = []
    train_original_losses = []
    train_reg_vals = []
    train_normalized_reg_vals = []
    train_grads_norms = []
    test_losses = []
    test_accuracies = []
    with open(log_path, 'a') as f:
        f.write(f"experiment_name={experiment_name} \n")
        print(f"Script arguments: \n")
        print(f"{args} \n")
        if args.finetuning:
            f.write(f"Finetuning mode: loaded_model_test_acc={100. * loaded_model_test_acc} \n")

    _, (jacobian_l1_mean, jacobian_l1_std), (jacobian_l2_mean, jacobian_l2_std), (jacobian_linf_mean, jacobian_linf_std) = eval_test_jacobian_norm(args, model, device, test_loader, args.reg_type)
    jacobian_norm_stats_before_training = {"jacobian_l1_mean": jacobian_l1_mean, "jacobian_l1_std": jacobian_l1_std,
                                          "jacobian_l2_mean": jacobian_l2_mean, "jacobian_l2_std": jacobian_l2_std,
                                          "jacobian_linf_mean": jacobian_linf_mean, "jacobian_linf_std": jacobian_linf_std}

    for epoch in range(1, args.epochs + 1):
        # adversarial training
        if args.AT_type == "Nat_test_time_jacobian":
            epoch_loss, epoch_original_loss, epoch_reg_val, epoch_grad_norm = test_time_jacobian_fintune_epoch(args, model, device, test_loader, optimizer, epoch)
        else:
            epoch_loss, epoch_original_loss, epoch_reg_val, epoch_grad_norm = train(args, model, device, train_loader, optimizer, epoch, args.AT_type)

        print('================================================================')
        if (epoch == args.epochs) or ((epoch == args.epochs-30 or epoch == 100) and args.AT_type == "TRADE"):
            if args.use_AdSS:
                print(f"{list(model.state_dict().keys())}")
                torch.save(model.state_dict(), os.path.join(model_dir, args.model_name+args.AT_type+'{}'.format(args.AdSS_Type)+ '-epoch{}.pt'.format(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(model_dir, args.model_name+args.AT_type+ '-epoch{}.pt'.format(epoch)))

        scheduler.step()
        test_loss, test_acc = eval_test(args, model, device, test_loader)

        train_losses.append(epoch_loss)
        train_original_losses.append(epoch_original_loss)
        train_reg_vals.append(epoch_reg_val)
        train_normalized_reg_vals.append(epoch_reg_val*args.alpha)
        train_grads_norms.append(epoch_grad_norm)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc*100)

        with open(log_path, 'a') as f:
            f.write(f"epoch:{epoch} lr={optimizer.param_groups[0]['lr']} epoch_loss={epoch_loss} epoch_original_loss={epoch_original_loss} epoch_reg_val={epoch_reg_val} epoch_grad_norm={epoch_grad_norm} test_acc={100*test_acc}\n")

        print(f"epoch:{epoch} lr={optimizer.param_groups[0]['lr']} epoch_loss={epoch_loss} epoch_original_loss={epoch_original_loss} epoch_reg_val={epoch_reg_val} epoch_grad_norm={epoch_grad_norm} test_acc={100*test_acc}")

    _, (jacobian_l1_mean, jacobian_l1_std), (jacobian_l2_mean, jacobian_l2_std), (jacobian_linf_mean, jacobian_linf_std) = eval_test_jacobian_norm(args, model, device, test_loader, args.reg_type)
    jacobian_norm_stats_after_training = {"jacobian_l1_mean": jacobian_l1_mean, "jacobian_l1_std": jacobian_l1_std,
                                           "jacobian_l2_mean": jacobian_l2_mean, "jacobian_l2_std": jacobian_l2_std,
                                           "jacobian_linf_mean": jacobian_linf_mean,
                                           "jacobian_linf_std": jacobian_linf_std}

    # Save metrics to .npz files
    np.savez(os.path.join(model_dir, "train_losses.npz"), train_losses=train_losses)
    np.savez(os.path.join(model_dir, "train_original_losses.npz"), train_original_losses=train_original_losses)
    np.savez(os.path.join(model_dir, "train_reg_vals.npz"), train_reg_vals=train_reg_vals)
    np.savez(os.path.join(model_dir, "train_normalized_reg_vals.npz"), train_normalized_reg_vals=train_normalized_reg_vals)
    np.savez(os.path.join(model_dir, "train_grads_norms.npz"), train_grads_norms=train_grads_norms)

    # plot curves
    epochs = range(0, len(train_losses))
    plot_all_loss_curves(model_dir=model_dir, experiment_name=experiment_name, epochs=epochs, train_losses=train_losses, train_original_losses=train_original_losses, train_normalized_reg_vals=train_normalized_reg_vals, train_reg_vals=train_reg_vals)
    plot_loss_curves(output_dir=model_dir, epochs=epochs, train_losses=train_losses, train_original_losses=train_original_losses, train_reg_vals=train_reg_vals, experiment_name=experiment_name)
    plot_accuracy_curves(output_dir=model_dir, epochs=epochs, test_accuracies=test_accuracies, experiment_name=experiment_name)
    plot_grads_norm_curve(output_dir=model_dir, epochs=epochs, grad_norms=train_grads_norms, experiment_name=experiment_name)

    print(f"jacobian_norm_stats before training: l1_mean={jacobian_norm_stats_before_training['jacobian_l1_mean']} l1_std={jacobian_norm_stats_before_training['jacobian_l1_std']} l2_mean={jacobian_norm_stats_before_training['jacobian_l2_mean']} l2_std={jacobian_norm_stats_before_training['jacobian_l2_std']} l_inf_mean={jacobian_norm_stats_before_training['jacobian_linf_mean']} l_inf_std={jacobian_norm_stats_before_training['jacobian_linf_std']}")
    print(f"jacobian_norm_stats after training: l1_mean={jacobian_norm_stats_after_training['jacobian_l1_mean']} l1_std={jacobian_norm_stats_after_training['jacobian_l1_std']} l2_mean={jacobian_norm_stats_after_training['jacobian_l2_mean']} l2_std={jacobian_norm_stats_after_training['jacobian_l2_std']} l_inf_mean={jacobian_norm_stats_after_training['jacobian_linf_mean']} l_inf_std={jacobian_norm_stats_after_training['jacobian_linf_std']}")
    with open(log_path, 'a') as f:
        f.write(f"jacobian_norm_stats before training: l1_mean={jacobian_norm_stats_before_training['jacobian_l1_mean']} l1_std={jacobian_norm_stats_before_training['jacobian_l1_std']} l2_mean={jacobian_norm_stats_before_training['jacobian_l2_mean']} l2_std={jacobian_norm_stats_before_training['jacobian_l2_std']} l_inf_mean={jacobian_norm_stats_before_training['jacobian_linf_mean']} l_inf_std={jacobian_norm_stats_before_training['jacobian_linf_std']} \n")
        f.write(f"jacobian_norm_stats after training: l1_mean={jacobian_norm_stats_after_training['jacobian_l1_mean']} l1_std={jacobian_norm_stats_after_training['jacobian_l1_std']} l2_mean={jacobian_norm_stats_after_training['jacobian_l2_mean']} l2_std={jacobian_norm_stats_after_training['jacobian_l2_std']} l_inf_mean={jacobian_norm_stats_after_training['jacobian_linf_mean']} l_inf_std={jacobian_norm_stats_after_training['jacobian_linf_std']} \n")
    print("train_trades_cifar10.py ended!")
    print(f"args={args}")
if __name__ == '__main__':
    main()