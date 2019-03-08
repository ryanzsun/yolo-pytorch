import torch.nn.init as init
import torch.nn as nn



def init_weights(m):
    if type(m) == nn.Conv3d :
        init.kaiming_uniform_(m.weight,nonlinearity='leaky_relu')