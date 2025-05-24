import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms
from autoattack import AutoAttack
from torch.autograd import Variable
from utils.args import Build_Parser
from utils.evalution import eval_test, adv_test, adv_eval_train, eval_train
from utils.cifar10 import build_model
from datetime import datetime

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

global global_noise_data
# Free Adversarial Training Module        
global_noise_data = torch.zeros([args.batch_size, 3, 32, 32]).cuda()

def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)

def adv_train(args, model, device, train_loader, optimizer, epoch):
    global global_noise_data
    criterion = nn.CrossEntropyLoss()
    # switch to train mode
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        input = data.to(device)
        target = target.to(device)
        for j in range(4):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            output = model(in1)
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, 4/255)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-4/255, 4/255)

            optimizer.step()
        if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))



def AA_adv_test(model,args,log_path):
    model.eval()
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
    
def main():
    start_time = datetime.now()
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # init model, ResNet18() can be also used here for training
    if args.use_AdSS:
        log_path = os.path.join(model_dir,'train_log'+args.model_name+'_AdSS_{}'.format(args.AdSS_Type)+args.AT_type+'.txt')
    else:    
        log_path = os.path.join(model_dir,'train_log'+args.model_name+args.AT_type+'.txt')

    log_file = open(log_path, 'w')
    model = build_model(args, args.model_name).to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    else:
        print("Using a single GPU")
        
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    for epoch in range(1, args.epochs + 1):
        # adversarial training
        adv_train(args, model, device, train_loader, optimizer, epoch)
        print('================================================================')
        if args.AT_type != 'Nat' and epoch % args.AA_lags == 0:
            AA_adv_test(model, args, log_path)
        # save checkpoint
        if epoch == args.epochs or epoch == 150:
            if args.use_AdSS:
                torch.save(model.state_dict(), os.path.join(model_dir, args.model_name+args.AT_type+'{}'.format(args.AdSS_Type)+ '-epoch{}.pt'.format(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(model_dir, args.model_name+args.AT_type+ '-epoch{}.pt'.format(epoch)))
        scheduler.step()

if __name__ == '__main__':
    main()