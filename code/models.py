import torch.nn as nn
import numpy as np 
from collections import OrderedDict


class TwoLayer(nn.Module):
    def __init__(self,
                 input_dim=100,
                 num_classes=2, 
                 in_channels=1,
                 out_channels=10,
                 kernel_size=3,
                 stride=1, 
                 conv2d=False,
                 mlp=True, 
                 bias=True,
                 linear=False, 
                 num_hidden_nodes=2
    ):
        super(TwoLayer, self).__init__()
        self.input_dim = input_dim
        self.num_channels=in_channels
        self.num_classes = num_classes
        self.num_out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.bias=bias
        self.num_hidden_nodes = num_hidden_nodes
        self.linear = linear
        
        if self.linear:
            activ = nn.Sequential()
        else:
            activ = nn.ReLU(True)

        if mlp:
            self.feature_extractor = nn.Sequential(OrderedDict([
                ('conv1', nn.Linear(self.input_dim, self.num_hidden_nodes)),
                ('relu1', activ)]))
            self.size_final = self.num_hidden_nodes
        
        else:                                 
            if conv2d:
                # 2D convolution 
                self.feature_extractor = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(self.num_channels, self.num_out_channels,
                                        self.kernel_size, self.stride,
                                        padding=0, 
                                        bias=bool(self.bias))),
                    ('relu1', activ)]))
                self.size_final = (self.input_dim - self.kernel_size)/self.stride + 1
                # Since 2D
                self.size_final = self.num_out_channels*int(self.size_final)*int(self.size_final)
                
            else:
                # 1D convolution 
                self.feature_extractor = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv1d(self.num_channels, self.num_out_channels,
                                        self.kernel_size, self.stride,
                                        padding=0,
                                        bias=bool(self.bias))),
                    ('relu1', activ)]))
                self.size_final = int(self.num_out_channels*((self.input_dim - self.kernel_size)/self.stride + 1))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.size_final, self.num_classes))]))

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, self.size_final))
        return logits


# class deepMLP(nn.Module):
#     def __init__(self,
#                  input_dim=100,
#                  num_classes=2, 
#                  bias=True,
#                  num_hidden_nodes=2,
#                  num_layers=3):
#         super(TwoLayer, self).__init__()

    
# class SmallCNN(nn.Module):
#     def __init__(self, drop=0.5, in_channels=1):
#         super(SmallCNN, self).__init__()

#         self.num_channels = in_channels
#         self.num_classes = 2

#         activ = nn.ReLU(True)

#         self.feature_extractor = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(self.num_channels, 32, 5, padding=2)),
#             ('relu1', activ),
#             ('conv2', nn.Conv2d(32, 32, 3)),
#             ('relu2', activ),
#             ('maxpool1', nn.MaxPool2d(2, 2)),
#             ('conv2', nn.Conv2d(32, 64, 5, padding=2)),
#             ('relu2', activ),
#             #('conv4', nn.Conv2d(64, 64, 3)),
#             #('relu4', activ),
#             ('maxpool2', nn.MaxPool2d(2, 2)),
#         ]))

#         self.classifier = nn.Sequential(OrderedDict([
#             #('fc1', nn.Linear(64*7*7, 1024)),
#             #('relu1', activ),
#             #('drop', nn.Dropout(drop)),
#             # ('fc2', nn.Linear(200, 200)),
#             # ('relu2', activ),
#             ('fc2', nn.Linear(64*7*7, self.num_classes)),
#         ]))

#     def __forward__(self, input):
#         #input = input.unsqueeze(1)
#         features = self.feature_extractor(input)
#         #print(features.shape)
#         logits = self.classifier(features.view(-1, 64*7*7))
#         return logits
