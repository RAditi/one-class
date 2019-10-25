import torch 
import torch.nn.functional as F
import torch.nn as nn

"""
Return norm of tensor
"""

def l2_norm(x):
    flattened = x.view(x.shape[0], -1)
    norm = torch.norm(flattened, p=2, dim=1)
    norm = norm.repeat(1, x.shape[1])
    return norm

"""
Function to compute adversarial example 
in l2 ball around positive example and assign negative label 
Assumes inputs are between 0 and 1 
Assumptions:
x_natural is flattened version 
"""

def one_class_adv_loss(model, 
                       x_natural, 
                       target, 
                       radius,
                       dist_lambda, 
                       step_size=0.1, 
                       num_gradient_steps=10): 
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # Assuming domain is [0, 1] and randomly initializing across domain
    x_adv = torch.randn(x_natural.shape).cuda().detach()

    for _ in range(num_gradient_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            new_targets = torch.ones(batch_size, 1)
            # Loss with respect to class 1 
            new_loss = F.cross_entropy(model(x_adv), new_targets)
            diff = x_adv - x_natural
            norm_diff = torch.sum(torch.mul(diff, diff), dim=1)
            dist_loss = torch.sum(norm_diff - radius*radius)
            # dist_loss tries to keep the distance close to R
            new_loss = new_loss + dist_lambda*dist_loss
            grad = torch.autograd.grad(new_loss, [x_adv])[0]
            # normalized gradient
            grad_normalized = torch.div(grad, l2_norm(grad))
            x_adv = x_adv.detach() + step_size*grad_normalized

    # Taking images to the R radius ball 
    for idx_batch in range(batch_size):
        eta_x_adv = x_adv[idx_batch] - x_natural[idx_batch]
        norm_eta = l2_norm(eta_x_adv)
        eta_x_adv = eta_x_adv * 2*radius / l2_norm(eta_x_adv)
        x_adv[idx_batch] = x_natural[idx_batch] + eta_x_adv
            
    adv_loss = F.cross_entropy(model(x_adv), new_targets)
    # Computing only for positive instances
    is_positive = targets == 1
    num_positive = sum(is_positive)
    # Returning average 
    final_adv_loss = adv_loss[is_positive].sum()/num_positive 
    return final_adv_loss
