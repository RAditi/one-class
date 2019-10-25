from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from datasets import *
from models import *
from analyze_activations import *
#from models.net_mnist import *
#from models.small_cnn import *
#from trades import trades_loss
torch.set_printoptions(precision=5)


parser = argparse.ArgumentParser(description='PyTorch Simple Training')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--num_hidden_nodes', type=int, default=2, metavar='N',
                    help='Num hidden nodes')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--mom', type=float, default=0.0, metavar='M',
                    help='momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_dir', default='./model-mnist-smallCNN',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--dataset', type=str, default='custom', metavar='N',
                    help='dataset: one of mnist and custom')
parser.add_argument('--num_train', type=int, default=2000, metavar='N',
                    help='Number of train examples')
parser.add_argument('--num_test', type=int, default=200, metavar='N',
                    help='Number of test examples')
parser.add_argument('--out_channels', type=int, default=2, metavar='N',
                    help='Out channels')
parser.add_argument('--kernel_size', type=int, default=3, metavar='N',
                    help='Kernel size')
parser.add_argument('--permute', type=int, default=0, metavar='N',
                    help='whether to permute (1 to permute)')
parser.add_argument('--stride', type=int, default=3, metavar='N',
                    help='Stride')
parser.add_argument('--bias', type=int, default=1, metavar='N',
                    help='Whether to use bias')
parser.add_argument('--mlp', dest='mlp', action='store_true')
parser.set_defaults(mlp=False)

args = parser.parse_args()
print(args)
# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())
#torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate classfication  loss
        data = data.to(torch.float)
        if args.dataset == 'mnist':
            data = data.unsqueeze(1)
        target = target.to(torch.long)
        logits = model(data)
        feats = model.feature_extractor(data)
        hidden_norm = torch.norm(feats)*torch.norm(feats) 
        hidden_norm = torch.sum(feats)
        l2_loss = args.lambda_l2*hidden_norm/args.batch_size
        loss = F.cross_entropy(logits, target) + l2_loss
        with torch.enable_grad():
            logits = model(data)
            f = logits[:, 0] - logits[:, 1]
        grad_pos = (torch.autograd.grad(f, [data]))

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.to(torch.float)
            if args.dataset == 'mnist':
                data = data.unsqueeze(1)
            target= target.to(torch.long)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader, logits=False):
    model.eval()
    test_loss = 0
    correct = 0
    conf_matrix = torch.zeros(3, 3)
    probability = 0
    total_pos = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.to(torch.float)
            if args.dataset == 'mnist':
                data = data.unsqueeze(1)
            target = target.to(torch.long)
            output = model(data)

            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            # Changing the pred function
            if args.dataset != 'seq':
                scores_pos = (output[:, 1] - output[:, 0])
                pred = (scores_pos > args.threshold).long()

            correct += pred.eq(target.view_as(pred)).sum().item()
            for t, p in zip(target, pred):
                conf_matrix[t, p] += 1
            probability = probability + torch.sum((torch.mul(torch.sigmoid(output[:, 1] - output[:, 0]), target.view_as(output[:, 0]).float())), 0)
            total_pos = total_pos + torch.sum(target)
            
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_accuracy = correct / len(test_loader.dataset)
    #print(conf_matrix)
    # print(output[0:20, :])
    # print(target[0:20])
    #print(probability/total_pos)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 0.90*args.epochs:
        lr = args.lr * 0.001
    if epoch >= 0.75*args.epochs:
        lr = args.lr * 0.01
    if epoch >= 0.5*args.epochs:
        lr = args.lr * 0.1
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, Net() can be also used here for training
    # model = SmallCNN().to(device)
    pos_classes=[0]
    neg_classes=[1]

    custom_params = {}
    if args.dataset == 'custom':
        custom_params['k'] = args.k
        custom_params['reps'] = args.reps
        custom_params['num_train'] = args.num_train
        custom_params['num_test'] = args.num_test
        custom_params['epsilon'] = args.epsilon
        custom_params['permute'] = args.permute
        
    train_loader = torch.utils.data.DataLoader(
        OneClass(
            dataset_name=args.dataset,
            custom_params=custom_params,
            dataset_folder='data',
            train=True,
            pos_classes=pos_classes,
            neg_classes=neg_classes,
            transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    normal_test_loader = torch.utils.data.DataLoader(
        OneClass(
            dataset_name=args.dataset,
            custom_params=custom_params,
            dataset_folder='data',
            train=False, 
            pos_classes=pos_classes,
            neg_classes=neg_classes,
            transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    sin_test_loader = torch.utils.data.DataLoader(
            OneClass(
            dataset_name=args.dataset,
            custom_params=custom_params,
            dataset_folder='data',
            train=False,
            test_type='sin', 
            pos_classes=pos_classes,
            neg_classes=neg_classes,
            transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    sq_test_loader = torch.utils.data.DataLoader(
        OneClass(
            dataset_name=args.dataset,
            custom_params=custom_params,
            dataset_folder='data',
            train=False,
            test_type='sq', 
            pos_classes=pos_classes,
            neg_classes=neg_classes,
            transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # orig_test_loader = torch.utils.data.DataLoader(
    #     OneClass('data', train=False,
    #              pos_classes=pos_classes,
    #              neg_classes=neg_classes, 
    #              transform=transforms.ToTensor(),
    #              patch_start=-1),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # patch_test_loader = torch.utils.data.DataLoader(
    #     OneClass('data', train=False,
    #              pos_classes=pos_classes,
    #              neg_classes=neg_classes, 
    #              transform=transforms.ToTensor(),
    #              patch_start=0, 
    #              patch_end=14),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.dataset == 'mnist':
        input_dim = 28
        conv2d = True
    elif args.dataset == 'custom':
        input_dim = 2*args.reps*args.k
        conv2d = False
    elif args.dataset == 'seq':
        input_dim = 75
        conv2d=False
        num_classes=3
        
    model = TwoLayer(input_dim=input_dim,
                     in_channels=1,
                     out_channels=args.out_channels,
                     kernel_size=args.kernel_size,
                     stride=args.stride, 
                     conv2d=conv2d,
                     bias=args.bias,
                     num_hidden_nodes=args.num_hidden_nodes,
                     num_classes=num_classes, 
                     mlp=args.mlp).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        if epoch == args.epochs: 
            # evaluation on natural examples
            print('======================== Train=======================================')
            eval_train(model, device, train_loader)
            #eval_test(model, device, orig_test_loader)
            print('======================== Normal ===================================')
            eval_test(model, device, normal_test_loader)
            print('======================== Sin =================================')
            eval_test(model, device, sin_test_loader)
            print('======================== Square =====================================')
            eval_test(model, device, sq_test_loader)
        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-nn-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-nn-checkpoint_epoch{}.tar'.format(epoch)))

            if args.dataset == 'custom' and False:
                analyze_activations(args.model_dir,
                                    epoch, 
                                    custom_params,
                                    args.out_channels,
                                    args.kernel_size,
                                    args.stride,
                                    args.bias,
                                    args.mlp,
                                    args.num_hidden_nodes)
            elif args.dataset == 'mnist':
                pass
                    # patch_test_loader = torch.utils.data.DataLoader(
                    #     OneClass('data', train=False,
                    #              pos_classes=pos_classes,
                    #              neg_classes=neg_classes, 
                    #              transform=transforms.ToTensor(),
                    #              patch_start=0, 
                    #              patch_end=14),
                    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
                    # eval_test(model, device, test_loader, logits=True)
                    
                    
if __name__ == '__main__':
    main()
