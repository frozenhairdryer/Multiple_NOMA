from os import error
from numpy.core.fromnumeric import argmax
import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.optim as optim
import numpy as cp
import matplotlib.pyplot as plt
import matplotlib
import datetime
import cupy as cp
import tikzplotlib

device = 'cuda' if torch.cuda.is_available() else 'cpu'