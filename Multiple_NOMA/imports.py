from os import error
from numpy.core.fromnumeric import argmax
import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
import datetime
import tikzplotlib
import miniball # for modradius calculation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    import cupy as cp
except:
    import numpy as cp