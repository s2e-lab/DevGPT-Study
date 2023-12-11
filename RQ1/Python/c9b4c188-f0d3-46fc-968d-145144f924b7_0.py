import torch
import torch.nn as nn

m = nn.Softmax(dim=1)
input = torch.randn(2, 3)
output = m(input)
