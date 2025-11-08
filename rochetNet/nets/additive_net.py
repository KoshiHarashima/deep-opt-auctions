from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


class Net(nn.Module):

    def __init__(self, config, mode):
        super(Net, self).__init__()
        self.config = config
        self.mode = mode
        
        # Initialize weights and biases
        self.w_init = nn.init.normal_
        b_init_range = self.config.net.b_init
        self.b_init = lambda x: nn.init.uniform_(x, b_init_range[0], b_init_range[1])
        
        self.build_net()

    def build_net(self):
        """
        Initializes network variables
        """

        num_items = self.config.num_items
        num_hidden_units = self.config.net.num_hidden_units       
        
        # Utility network weights and bias
        self.alpha = nn.Parameter(torch.empty(num_items, num_hidden_units))
        self.w_init(self.alpha)
        
        self.bias = nn.Parameter(torch.empty(num_hidden_units))
        self.b_init(self.bias)

    def inference(self, x):
        """
        Inference
        """
        # Ensure x is a torch tensor and move to the same device as the model
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Move input to the same device as model parameters
        x = x.to(self.alpha.device)

        # Pad: add one more column and row
        # padding_w: [[0, 0], [0, 1]] -> pad last dimension with 0 on left, 1 on right
        # padding_b: [[0, 1]] -> pad with 0 on left, 1 on right
        w = F.pad(torch.sigmoid(self.alpha), (0, 1), mode='constant', value=0)
        b = F.pad(self.bias, (0, 1), mode='constant', value=0)

        utility = torch.matmul(x, w) + b
        U = F.softmax(utility * self.config.net.eps, dim=-1)
        
        if self.mode == "train":
            a = torch.matmul(U, w.t())
        else:
            # One-hot encoding of argmax
            max_indices = torch.argmax(utility, dim=-1)
            one_hot = F.one_hot(max_indices, num_classes=self.config.net.num_hidden_units + 1).float()
            one_hot = one_hot.to(self.alpha.device)
            a = torch.matmul(one_hot, w.t())
        
        p = torch.sum(a * x, dim=-1) - torch.max(utility, dim=-1)[0]
        
        return a, p
