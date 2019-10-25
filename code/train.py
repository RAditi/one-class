import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import compute_margin 
from loss import one_class_adv_loss
"""
args contains: 
args.lr 
args.mom
args.opt
args.epochs
args.log_interval
args.lamda
args.one_class_adv
args.radius
args.dist_lambda
args.step_size
args.num_gradient_steps
args.return_margin
"""
def train(args, 
          model, 
          device, 
          train_loader, 
          print_progress=False):

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.mom)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               momentum=args.mom)
    if args.return_margin:
        MARGIN = []
    if args.one_class_adv: 
        ADV_LOSS = []
    LOSS = []    
    for epoch in range(args.epochs): 
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # calculate classfication  loss
            data = data.to(torch.float)
            target = target.to(torch.long)
            target = torch.squeeze(target)
            logits = model(data)
            clf_loss = F.cross_entropy(logits, target)
            if args.one_class_adv: 
                adv_loss = one_class_adv_loss(model, data, target,
                                              args.radius,
                                              args.dist_lambda, 
                                              args.step_size,
                                              args.num_gradient_steps)
                loss = clf_loss + adv_loss*args.lamda
            else: 
                loss = clf_loss
            loss.backward()
            optimizer.step()
            # print progress
            if batch_idx % args.log_interval == 0:
                if args.return_margin:
                    margin = compute_margin(model, 
                                            device, 
                                            train_loader)
                    MARGIN.append(margin)
                if args.one_class_adv: 
                    ADV_LOSS.append(adv_class.item())
                    
                LOSS.append(loss.item())
                if print_progress: 
                    print('Train Epoch: {}\tLoss: {:.6f}'.format(
                        epoch, loss.item()))
                    if args.return_margin:
                        print('Margin at Epoch {}: {:.6f}'.format(epoch, margin))
                    if args.return_margin:
                        print('Adversarial loss at Epoch {}: {:.6f}'.format(epoch, adv_class))
    
    if args.return_margin: 
        return MARGIN, LOSS
    if args.one_class_adv: 
        return ADV_LOSS, LOSS
    else return LOSS
    
