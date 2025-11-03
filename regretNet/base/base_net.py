from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_summary(x):
    """ 
    Helper for activation summaries (placeholder for compatibility).
    In PyTorch, logging can be handled via TensorBoard or custom logging.
    Args:
        x: Tensor
    """
    # PyTorch doesn't have built-in summary ops like TF1
    # This function is kept for API compatibility but does nothing
    pass


class BaseNet(nn.Module):
    
    def __init__(self, config):
        super(BaseNet, self).__init__()
        self.config = config
        
        """ Set initializer """
        if self.config.net.init == 'None':
            self.init = None
        elif self.config.net.init == 'gu':
            self.init = nn.init.xavier_uniform_
        elif self.config.net.init == 'gn':
            self.init = nn.init.xavier_normal_
        elif self.config.net.init == 'hu':
            self.init = nn.init.kaiming_uniform_
        elif self.config.net.init == 'hn':
            self.init = nn.init.kaiming_normal_
        else:
            self.init = None
        
        if self.config.net.activation == 'tanh':
            self.activation = torch.tanh
        elif self.config.net.activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.relu  # default
               
    def build_net(self):
        """
        Initializes network variables
        """
        raise NotImplementedError
        
    def inference(self, x):
        """ 
        Inference 
        """
        raise NotImplementedError
