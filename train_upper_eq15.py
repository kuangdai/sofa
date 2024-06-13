import os
import sys

import torch

ns = torch.cat((torch.arange(10, 100, 10), torch.arange(100, 3001, 100), torch.arange(3500, 10001, 500)))
for n in ns:
    os.system(f"python train_upper.py -a {n} -N eq15_{n} -s {sys.argv[1]}")