import torch
import torch.nn.functional as F
from utils.used_attacks import PGD
from autoattack.autopgd_base import APGDAttack as APGD

def eval_train(args, model, device, train_loader):
    model.eval()
    
    train_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
                
    train_loss /= len(train_loader.dataset)

    print('Train: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    train_loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    
    return train_loss, training_accuracy

def adv_eval_train(args, model, device, train_loader, attack_method = 'PGD'):
    model.eval()
    
    train_loss = 0
    correct = 0
    
    if attack_method == 'PGD':
        attack = PGD(model, args.epsilon, args.step_size, args.num_steps, normalize=False)
    elif args.attack_type =='APGD':
        attack = APGD(model, eps=args.epsilon, seed=args.seed,device=device)

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        if args.attack_type =='APGD':
            data_adv = attack.perturb(data, target)
        else:
            data_adv = attack(data, target)
        with torch.no_grad():
            output = model(data_adv)
                    
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
                
    train_loss /= len(train_loader.dataset)

    print('Adv Train: Average loss: {:.4f}, Adv Accuracy: {}/{} ({:.0f}%)'.format(
    train_loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    
    return train_loss, training_accuracy

def adv_test(args, model, device, test_loader, attack_method = 'PGD'):
    model.eval()

    test_loss = 0
    correct = 0
    if attack_method == 'PGD':
        attack = PGD(model, args.epsilon, args.step_size, args.num_steps, normalize=False)
    elif args.attack_type =='APGD':
        attack = APGD(model, eps=args.epsilon, seed=args.seed,device=device)
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if args.attack_type =='APGD':
            data_adv = attack.perturb(data, target)
        else:
            data_adv = attack(data, target)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            output = model(data_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            torch.cuda.empty_cache()
            
    test_loss /= len(test_loader.dataset)

    print('Adv Test: Average loss: {:.4f}, Adv Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            print(f"DEBUG: eval_test: output={output}")
            test_loss += F.cross_entropy(output, target, size_average=False).item()
                
            pred = output.max(1, keepdim=True)[1]
            print(f"DEBUG: eval_test: pred={pred}")
            correct += pred.eq(target.view_as(pred)).sum().item()
        torch.cuda.empty_cache()
           
    test_loss /= len(test_loader.dataset)

    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy



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
        #print(f"DEBUG: eval_test_jacobian_norm: output={output}")
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
    