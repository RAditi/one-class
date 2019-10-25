import numpy as np
import matplotlib.pyplot as plt
from dataset import CustomDataset
import torch
LARGE = 10000
"""
Visualizing a two dimensional dataset
"""
def visualize_dataset(X, y):
    colormap = np.array(['r', 'b'])
    y = y.astype(int)
    plt.scatter(X[:, 0], X[:, 1], c=colormap[np.ravel(y)])
    return plt


def visualize_classifier(model, device, min_x, max_x, min_y, max_y, test_loader, **kwargs):
    x1 = np.linspace(min_x, max_x, 100)
    x2 = np.linspace(min_y, max_y, 100)
    X1, X2 = np.meshgrid(x1, x2)
    data_matrix = np.concatenate([np.reshape(X1, [-1, 1]), np.reshape(X2, [-1, 1])], axis = 1)
    model.eval()
    y = np.random.rand(100*100, 1)
    vis_dataset = CustomDataset(data_matrix, y)

    vis_loader = torch.utils.data.DataLoader(
        vis_dataset,
        batch_size=100,
        shuffle=False,
        **kwargs)

    preds = []
    #data_matrix = []
    with torch.no_grad():
        for data, target in vis_loader:
            data, target = data.to(device), target.to(device)
            data = data.to(torch.float)
            target= target.to(torch.long)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cpu().numpy()
            preds.append(pred)
            #data_matrix.append(data.cpu().numpy())

    preds = np.asarray(preds)
    preds = np.ravel(preds)
    #data_matrix = np.asarray(data_matrix)
    #data_matrix = np.reshape(data_matrix, [-1, 2])
    plt = visualize_dataset(data_matrix, preds)
    return plt, data_matrix, preds
    
"""
Currently implemented only for two layer networks
"""
def compute_margin(model, device, data_loader):
    model.eval()
    weight_1 = model.feature_extractor[0].weight
    bias_1 = model.feature_extractor[0].bias
    weight_2 = model.classifier[0].weight
    bias_2 = model.classifier[0].bias
    if model.linear:
        final_lin = torch.mm(torch.transpose(weight_1, 0, 1), torch.transpose(weight_2, 0, 1))
        final_lin = final_lin[:, 0] - final_lin[:, 1]
        final_lin = final_lin.view([2, 1])
        final_bias = torch.mm(bias_1.view([1, -1]), torch.transpose(weight_2, 0, 1))
        final_bias = final_bias[0, 0] - final_bias[0, 1] + bias_2[0] - bias_2[1]

    minimum_margin = LARGE 
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        data = data.to(torch.float)
        target = target.to(torch.float)
        
        current_margin = torch.mm(data, final_lin)
        current_margin = -1*torch.mul(current_margin, 2*target - 1) + final_bias
        current_margin = torch.min(current_margin).cpu().detach().numpy()
        minimum_margin = np.minimum(minimum_margin, current_margin)
    return minimum_margin
