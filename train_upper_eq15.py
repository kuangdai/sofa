import os

import torch

ns = torch.cat((torch.arange(10, 100, 10), torch.arange(100, 3001, 100)))
for n in ns:
    os.system(f"python train_upper.py -a {n} -N eq15_{n}")
