import torch
import torch.nn as nn

s = torch.tensor([[1,2],
                  [3,4]])
print(s.split(1, dim=0))