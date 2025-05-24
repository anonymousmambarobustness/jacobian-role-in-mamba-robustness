import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from utils.used_attacks import PGD
from autoattack.autopgd_base import APGDAttack as APGD
import time
import math
if os.getenv('CONDA_DEFAULT_ENV') == "exploring_mamba_env7":
    from mamba_ssm.modules.mamba_simple import Mamba
def compute_jacobian_penalty(model, inputs):
    batch_size = inputs.shape[0]
    batched_inputs = inputs.clone().requires_grad_()
    batched_outputs = model(batched_inputs)
    output_dim = batched_outputs.shape[-1]
    jacobian_rows = []
    cotangents = torch.eye(output_dim, device=batched_inputs.device)
    for i in range(output_dim):
        cotangent = cotangents[i].repeat(batch_size, 1)
        jacobian_row = torch.autograd.grad(
            outputs=batched_outputs,
            inputs=batched_inputs,
            grad_outputs=cotangent,
            create_graph=True,
            retain_graph=True,
        )[0]
        jacobian_rows.append(jacobian_row)

    jacobian = torch.stack(jacobian_rows, dim=1) # (batch_size, output_dim, input_dim)
    jacobian_norm = jacobian.view(batch_size, -1).norm(2, 1)
    return torch.mean(jacobian_norm ** 2)


def compute_gp(inputs, outputs):
    batch_size = inputs.shape[0]
    output_dim = outputs.shape[-1]
    jacobian_rows = []
    cotangents = torch.eye(output_dim, device=inputs.device)
    for i in range(output_dim):
        cotangent = cotangents[i].repeat(batch_size, 1)
        jacobian_row = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=cotangent,
            create_graph=True,
            retain_graph=True,
        )[0]
        jacobian_rows.append(jacobian_row)

    jacobian = torch.stack(jacobian_rows, dim=1) # (batch_size, output_dim, input_dim)
    jacobian_norm = jacobian.view(batch_size, -1).norm(2, 1)
    return torch.mean(jacobian_norm ** 2)


def compute_gp_correct_class(inputs, outputs, targets):
    print(f"DEBUG: compute_gp_correct_class: inputs.shape={inputs.shape}")
    batch_size = inputs.shape[0]
    output_dim = outputs.shape[-1]
    cotangents = torch.zeros(batch_size, output_dim, device=inputs.device)
    cotangents[range(batch_size), targets] = 1.0
    jacobian_row = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=cotangents,
        create_graph=True,
        retain_graph=True,
    )[0]
    jacobian_norm = jacobian_row.view(batch_size, -1).norm(2, 1)
    print(f"DEBUG: compute_gp_correct_class: jacobian_row.shape={jacobian_row.shape}")
    return torch.mean(jacobian_norm ** 2)


def compute_gp_correct_class_l2(inputs, outputs, targets):
    print(f"DEBUG: compute_gp_correct_class_l2: inputs.shape={inputs.shape}")
    batch_size = inputs.shape[0]
    output_dim = outputs.shape[-1]
    cotangents = torch.zeros(batch_size, output_dim, device=inputs.device)
    cotangents[range(batch_size), targets] = 1.0
    jacobian_row = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=cotangents,
        create_graph=True,
        retain_graph=True,
    )[0]
    jacobian_norm = jacobian_row.view(batch_size, -1).norm(2, 1)
    print(f"DEBUG: compute_gp_correct_class_l2: jacobian_row.shape={jacobian_row.shape}")
    return torch.mean(jacobian_norm)


def compute_gp_correct_class_l1(inputs, outputs, targets):
    print(f"DEBUG: compute_gp_correct_class_l1: inputs.shape={inputs.shape}")
    batch_size = inputs.shape[0]
    output_dim = outputs.shape[-1]
    cotangents = torch.zeros(batch_size, output_dim, device=inputs.device)
    cotangents[range(batch_size), targets] = 1.0
    jacobian_row = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=cotangents,
        create_graph=True,
        retain_graph=True,
    )[0]
    jacobian_norm = jacobian_row.view(batch_size, -1).norm(1, 1) # (p=1,dim=1)
    print(f"DEBUG: compute_gp_correct_class_l1: jacobian_row.shape={jacobian_row.shape}")
    return torch.mean(jacobian_norm)


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
        create_graph=True,
        retain_graph=True,
    )[0]
    jacobian_norm = jacobian_row.view(batch_size, -1).norm(2, 1)
    return torch.mean(jacobian_norm ** 2)


def compute_spectral_norm(weight, n_power_iterations=1, eps=1e-12):
    weight = weight.squeeze()
    # If the weight is 1D, treat it as a vector, return its norm directly
    if len(weight.shape) == 1:
        return torch.linalg.norm(weight, ord=2)  # Spectral norm is just the 2-norm for 1D vectors
    if len(weight.shape) > 2:
        weight = weight.reshape(weight.shape[0], -1)

    # For 2D and 3D weights, perform power iteration
    b_k = torch.randn(weight.shape[1], dtype=weight.dtype, device=weight.device)

    for _ in range(n_power_iterations):
        b_k1 = torch.matmul(weight, b_k)
        b_k1_norm = torch.linalg.norm(b_k1) + eps  # Avoid division by zero
        b_k = b_k1 / b_k1_norm

    return torch.dot(b_k, torch.matmul(weight, b_k)) # (Rayleigh quotient)
    #return torch.norm(torch.matmul(weight, b_k), p=2)  # Final spectral norm estimate


def apply_spectral_norm_to_model(model, n_power_iterations=1):
    """
    Applies spectral normalization to each weight tensor in the model.

    Args:
        model (torch.nn.Module): The model whose weights need to be normalized.
        n_power_iterations (int): Number of power iterations to approximate the spectral norm.
    """
    for name, param in model.named_parameters():
        print(f"apply_spectral_norm_to_model: param_name={name} param.shape={param.shape}")
        if len(param.shape) == 1:  # 1D weights
            sn = compute_spectral_norm(param, n_power_iterations)
            param.data /= sn  # Normalize the 1D weight tensor

        elif len(param.shape) == 2:  # 2D weights (e.g., Linear layers)
            sn = compute_spectral_norm(param, n_power_iterations)
            param.data /= sn  # Normalize the 2D weight tensor

        elif len(param.shape) == 3:  # 3D weights (e.g., Conv layers)
            for i in range(param.shape[0]):  # Iterate over the batch or channel dimension
                sn = compute_spectral_norm(param[i], n_power_iterations)
                param[i].data /= sn  # Normalize each slice

def compute_self_attention_lip_bound(V,K,Q): # V.shape=(batch_size, L, 2*d_model) K.shape=Q.shape=(batch_size, L, d_state)
    #print(f"compute_self_attention_lip_bound: (X)V.shape={V.shape} (B)K.shape={K.shape} (C)Q.shape={Q.shape}")
    n = V.shape[1] #seq_length 'L'
    k = n
    squared_norms = torch.sum(V ** 2, dim=-1)
    #print(f"compute_self_attention_lip_bound: squared_norms.shape={squared_norms.shape}")
    mean_squared_norms = squared_norms.mean(dim=-1)
    #print(f"compute_self_attention_lip_bound: mean_squared_norms.shape={mean_squared_norms.shape}")
    R = torch.sqrt(mean_squared_norms) # R.shape=(batch_size,)
    K_transpose = torch.transpose(K,dim0=1,dim1=2)
    #print(f"compute_self_attention_lip_bound: R.shape={R.shape} K_transpose.shape={K_transpose.shape} Q.shape={Q.shape}")
    #print(f"(K_transpose@Q).shape={(K_transpose@Q).shape} k={k}")
    A = (K_transpose@Q) / math.sqrt(k)
    #print(f"compute_self_attention_lip_bound: A.shape={A.shape}")
    A_norm_squared = torch.sum(A**2,dim=(1,2))
    term = (A_norm_squared*(R**4)*(n+1) + n)
    V_norm = torch.norm(V,p=2,dim=(1,2))
    #print(f"compute_self_attention_lip_bound: A_norm_squared.shape={A_norm_squared.shape} V_norm.shape={V_norm.shape} term.shape={term.shape}")
    upper_bound = math.sqrt(3)*V_norm*(term**0.5)
    #print(f"upper_bound.shape={upper_bound.shape} torch.mean(upper_bound).shape={torch.mean(upper_bound).shape}")
    return torch.mean(upper_bound)




def regularization_term(inputs, outputs, partial_inputs, partial_outputs, targets, regularization_type, model, args):
    if regularization_type == "none":
        return 0
    elif regularization_type == "gp":
        return compute_gp(inputs=inputs, outputs=outputs)
    elif regularization_type == "gp_correct_class" or regularization_type == "adv_gp_correct_class":
        return compute_gp_correct_class(inputs=inputs, outputs=outputs, targets=targets)
    elif regularization_type == "gp_correct_class_l1":
        return compute_gp_correct_class_l1(inputs=inputs, outputs=outputs, targets=targets)
    elif regularization_type == "gp_correct_class_l2":
        return compute_gp_correct_class_l2(inputs=inputs, outputs=outputs, targets=targets)
    elif regularization_type == "gp_correct_class_100_first_words":
        return compute_gp_correct_class(inputs=partial_inputs, outputs=partial_outputs, targets=targets)
    elif regularization_type == "gp_correct_class_100_last_words":
        print(f"DEBUG: regularization_term: gp_correct_class_100_last_words inputs.shape={inputs.shape} partial_outputs.shape={partial_outputs.shape}")
        return compute_gp_correct_class(inputs=inputs, outputs=partial_outputs, targets=targets)
    elif regularization_type == "gp_max_output":
        return  compute_gp_max_output(inputs=inputs, outputs=outputs)
    elif regularization_type == "jp":
        return compute_jacobian_penalty(model=model, inputs=inputs)
    elif regularization_type == "spectral":
        print(f"DEBUG: sum(model.module.layers_reg_vals)={sum(model.module.layers_reg_vals) if isinstance(model, torch.nn.DataParallel) else sum(model.layers_reg_vals)}")
        return sum(model.module.layers_reg_vals) if isinstance(model, torch.nn.DataParallel) else sum(model.layers_reg_vals)
    elif regularization_type == "spectral_norm":
        model = model.module if isinstance(model, torch.nn.DataParallel) else model
        for index, layer in enumerate(model.mapping_layers):
            if os.getenv('CONDA_DEFAULT_ENV') == "exploring_mamba_env7" and isinstance(layer, Mamba):
                scale_factor = args.alpha * model.layers_reg_vals[index]
                print(f"DEBUG: spectral_norm: scaling layer{index} parameters by {scale_factor}")
                for param_name, param in layer.named_parameters(): #TODO: check with Edo if no_grad() is required here
                    updated_param = param / scale_factor # Perform out-of-place scaling
                    param_name_parts = param_name.split('.')
                    submodule = layer
                    for part in param_name_parts[:-1]:  # Traverse the submodule hierarchy
                        submodule = getattr(submodule, part)

                    final_param_name = param_name_parts[-1]
                    print(f"DEBUG: final_param_name={final_param_name}")
                    setattr(submodule, final_param_name, torch.nn.Parameter(updated_param))  # Reassign the parameter
        return 0
    elif regularization_type == "attention_lip_bound":
        model = model.module if isinstance(model, torch.nn.DataParallel) else model
        reg_val = 0
        for idx, (V, K, Q) in enumerate(zip(model.V_list, model.K_list, model.Q_list)):
            current_lip_bound = compute_self_attention_lip_bound(V=V,K=K,Q=Q)
            #print(f"DEBUG: regularization_term: attention_lip_bound: layer{idx} lip_bound={current_lip_bound}")
            reg_val += current_lip_bound
        return reg_val
    else:
        print(f"regularization_term: reg_type={regularization_type} is not supported!")
        exit(1)

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(args,
                model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    if args.reg_type in ["gp", "gp_old", "gp_correct_class", "gp_correct_class_l1", "gp_correct_class_l2", "gp_max_output"]:  # for gradient penalty regularization we compute the gradient of the output with respect to the input
        x_natural.requires_grad = True
    elif args.reg_type == "adv_gp_correct_class":
        print(f"trades: adv_gp_correct_class")
        x_adv.requires_grad = True
    elif args.reg_type == "nat_adv_gp_correct_class":
        x_natural.requires_grad = True
        x_adv.requires_grad = True

    # calculate robust loss
    logits = model(x_natural)
    adv_logits = model(x_adv)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),F.softmax(logits, dim=1))
    loss = loss_natural + beta * loss_robust
    batch_original_loss = loss.item()
    if args.reg_type == "adv_gp_correct_class":
        adv_reg_val = regularization_term(inputs=x_adv, outputs=adv_logits, partial_inputs=None, partial_outputs=None, targets=y, regularization_type=args.reg_type, model=model, args=args)
        loss = loss + (args.alpha * adv_reg_val)
        batch_reg_val = adv_reg_val.item()
    elif args.reg_type == "nat_adv_gp_correct_class":
        adv_reg_val = regularization_term(inputs=x_adv, outputs=adv_logits, partial_inputs=None, partial_outputs=None, targets=y, regularization_type=args.reg_type, model=model, args=args)
        reg_val = regularization_term(inputs=x_natural, outputs=logits, partial_inputs=None, partial_outputs=None, targets=y, regularization_type=args.reg_type, model=model, args=args)
        print(f"nat_adv_gp_correct_class is not supported yet! -> exit program")
        exit(1)
    elif args.reg_type in ["gp", "gp_old", "gp_correct_class", "gp_correct_class_l1", "gp_correct_class_l2", "gp_max_output"]:
        reg_val = regularization_term(inputs=x_natural, outputs=logits, partial_inputs=None, partial_outputs=None, targets=y, regularization_type=args.reg_type, model=model, args=args)
        loss = loss + (args.alpha * reg_val)
        batch_reg_val = reg_val.item()
    elif args.reg_type == "none":
        batch_reg_val = 0


    batch_loss = loss.item()

    if args.reg_type in ["gp", "gp_old", "gp_correct_class", "gp_correct_class_l1","gp_correct_class_l2","gp_max_output"]:  # for gradient penalty regularization we compute the gradient of the output with respect to the input
        x_natural.requires_grad = False
    elif args.reg_type == "adv_gp_correct_class":
        x_adv.requires_grad = False
    elif args.reg_type == "nat_adv_gp_correct_class":
        x_natural.requires_grad = False
        x_adv.requires_grad = False

    return loss, batch_original_loss, batch_loss, batch_reg_val


def Nat_loss(args, model, data, target, criterion, mode):
    partial_data = None
    partial_out = None
    if args.reg_type in ["gp", "gp_old", "gp_correct_class", "gp_correct_class_l1", "gp_correct_class_l2", "gp_max_output"]:  # for gradient penalty regularization we compute the gradient of the output with respect to the input
        data.requires_grad = True

    out = model(data)
    loss = criterion(out, target)
    regularization_term_value = regularization_term(inputs=data, outputs=out, partial_inputs=partial_data, partial_outputs=partial_out, targets=target, regularization_type=args.reg_type, model=model, args=args)
    original_loss = loss.item()
    if args.reg_type != "none":
        loss = loss + (args.alpha * regularization_term_value)
    batch_loss = loss.item()
    batch_original_loss = original_loss
    batch_reg_val = regularization_term_value.item() if args.reg_type not in ["none","spectral_norm"] else regularization_term_value

    if args.reg_type in ["gp", "gp_old", "gp_correct_class", "gp_correct_class_l1", "gp_correct_class_l2", "gp_max_output", "gp_correct_class_100_last_words"]:  # for gradient penalty regularization we compute the gradient of the output with respect to the input
        data.requires_grad = False

    return loss, batch_original_loss, batch_loss, batch_reg_val


def get_auxiliary_mode(model_name):
    if "auxiliary_with_detach" in  model_name:
        return "auxiliary_with_detach"
    elif "auxiliary" in model_name:
        return "auxiliary"
    else:
        return None

def train(args, model, device, train_loader, optimizer, epoch, type='Madry'):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    epoch_original_loss = 0
    epoch_reg_val = 0
    total_grad_norm = 0
    num_batches = len(train_loader)
    mode = get_auxiliary_mode(args.model_name)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.reg_type == "official_spectral_norm":
            apply_spectral_norm_to_model(model=model, n_power_iterations=3)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # calculate robust loss
        if type == 'Madry':
            print(f"Entered type == 'Madry'!")
            #exit(1)
            criterion = nn.CrossEntropyLoss()
            if args.attack_type =='PGD':
                attack = PGD(model, args.epsilon, args.step_size, args.num_steps, normalize=False)
            elif args.attack_type =='APGD':
                attack = APGD(model, n_iter=20, eps=args.epsilon, seed=args.seed,device=device)
            if args.attack_type =='APGD':
                data_adv = attack.perturb(data, target)
            else:
                data_adv = attack(data, target)
     
            out = model(data_adv)
            loss = criterion(out, target)
            batch_original_loss = batch_loss = loss.item()
            batch_reg_val = 0
                       
        elif type == 'TRADE':
            out = None
            loss, batch_original_loss, batch_loss, batch_reg_val = trades_loss(args=args,
                                                                               model=model,
                                                                               x_natural=data,
                                                                               y=target,
                                                                               optimizer=optimizer,
                                                                               step_size=args.step_size,
                                                                               epsilon=args.epsilon,
                                                                               perturb_steps=args.num_steps,
                                                                               beta=args.beta)
        elif type == 'Nat' or type == 'Nat_test_set':
            criterion = nn.CrossEntropyLoss()
            loss, batch_original_loss, batch_loss, batch_reg_val = Nat_loss(args, model, data, target, criterion, mode)

        epoch_loss += batch_loss
        epoch_original_loss += batch_original_loss
        epoch_reg_val += batch_reg_val

        loss.backward()

        batch_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
        total_grad_norm += batch_grad_norm.item()

        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item(),time.time()-start_time))

    # Calculate and print average losses for the epoch
    epoch_loss = epoch_loss / num_batches
    epoch_original_loss = epoch_original_loss / num_batches
    epoch_reg_val = epoch_reg_val / num_batches
    epoch_grad_norm = total_grad_norm / num_batches
    return epoch_loss, epoch_original_loss, epoch_reg_val, epoch_grad_norm


def test_time_jacobian_fintune_epoch(args, model, device, test_loader, optimizer, epoch):
    model.train()
    start_time = time.time()
    epoch_loss = 0
    epoch_original_loss = 0
    epoch_reg_val = 0
    total_grad_norm = 0
    num_batches = len(test_loader)
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        data.requires_grad = True
        out = model(data)
        loss = regularization_term(inputs=data, outputs=out, partial_inputs=None, partial_outputs=None, targets=target, regularization_type=args.reg_type, model=model, args=args)
        epoch_loss += loss.item()
        loss.backward()
        batch_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
        total_grad_norm += batch_grad_norm.item()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}'.format(epoch, batch_idx * len(data),len(test_loader.dataset),100. * batch_idx / len(test_loader), loss.item(),time.time() - start_time))

    # Calculate and print average losses for the epoch
    epoch_loss = epoch_loss / num_batches
    epoch_original_loss = epoch_original_loss / num_batches
    epoch_reg_val = epoch_reg_val / num_batches
    epoch_grad_norm = total_grad_norm / num_batches
    return epoch_loss, epoch_original_loss, epoch_reg_val, epoch_grad_norm