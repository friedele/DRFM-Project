import torch 
import pandas as pd
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lbfgs_modify import LBFGSModify
from LBFGS import LBFGS
import torch.nn.functional as F
import math
import time
import numpy as np


test_acc_values=[43, 34, 47, 50, 55, 60]
test_loss_values=[1.5888177156448364, 1.6180187463760376, 1.2002453804016113, 1.3, 1.2, 1]
training_acc_values=[0, 45, 52, 55, 60, 65]
training_loss_values=[2.335207223892212, 1.7347087860107422, 1.4468894004821777, 1.40, 1.3, 1]

num_cols=len(test_acc_values)  
filename = "c:/users/friedele/Repos/DRFM/image_test_data.csv"
data_vectors = np.array([np.array([training_loss_values]).T,
             np.array([test_loss_values]).T,
             np.array([training_acc_values]).T,
             np.array([test_acc_values]).T])
df = pd.DataFrame(data_vectors.reshape(4,num_cols))
df.to_csv(filename, index=False)


