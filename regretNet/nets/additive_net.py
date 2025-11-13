from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import *


class Net(BaseNet):

    def __init__(self, config):
        super(Net, self).__init__(config)
        self.build_net()

    def build_net(self):
        """
        Initializes network variables
        """

        num_agents = self.config.num_agents
        num_items = self.config.num_items

        num_a_layers = self.config.net.num_a_layers        
        num_p_layers = self.config.net.num_p_layers

        num_a_hidden_units = self.config.net.num_a_hidden_units
        num_p_hidden_units = self.config.net.num_p_hidden_units
        
        # Alloc network weights and biases
        self.w_a = nn.ParameterList()
        self.b_a = nn.ParameterList()

        # Pay network weights and biases
        self.w_p = nn.ParameterList()
        self.b_p = nn.ParameterList()

        num_in = num_agents * num_items

        # Allocation network layers
        # Input Layer
        w_a_0 = nn.Parameter(torch.empty(num_in, num_a_hidden_units))
        if self.init is not None:
            self.init(w_a_0)
        self.w_a.append(w_a_0)

        # Hidden Layers
        for i in range(1, num_a_layers - 1):
            w_a_i = nn.Parameter(torch.empty(num_a_hidden_units, num_a_hidden_units))
            if self.init is not None:
                self.init(w_a_i)
            self.w_a.append(w_a_i)
                
        # Output Layer
        w_a_out = nn.Parameter(torch.empty(num_a_hidden_units, (num_agents + 1) * (num_items + 1)))
        if self.init is not None:
            self.init(w_a_out)
        self.w_a.append(w_a_out)

        # Biases
        for i in range(num_a_layers - 1):
            b_a_i = nn.Parameter(torch.zeros(num_a_hidden_units))
            self.b_a.append(b_a_i)
                
        b_a_out = nn.Parameter(torch.zeros((num_agents + 1) * (num_items + 1)))
        self.b_a.append(b_a_out)

        # Payment network layers
        # Input Layer
        w_p_0 = nn.Parameter(torch.empty(num_in, num_p_hidden_units))
        if self.init is not None:
            self.init(w_p_0)
        self.w_p.append(w_p_0)

        # Hidden Layers
        for i in range(1, num_p_layers - 1):
            w_p_i = nn.Parameter(torch.empty(num_p_hidden_units, num_p_hidden_units))
            if self.init is not None:
                self.init(w_p_i)
            self.w_p.append(w_p_i)
                
        # Output Layer
        w_p_out = nn.Parameter(torch.empty(num_p_hidden_units, num_agents))
        if self.init is not None:
            self.init(w_p_out)
        self.w_p.append(w_p_out)

        # Biases
        for i in range(num_p_layers - 1):
            b_p_i = nn.Parameter(torch.zeros(num_p_hidden_units))
            self.b_p.append(b_p_i)
                
        b_p_out = nn.Parameter(torch.zeros(num_agents))
        self.b_p.append(b_p_out)

    def inference(self, x):
        """
        Inference 
        """
        # Ensure x is a torch tensor and move to the same device as the model
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Move input to the same device as model parameters
        x = x.to(next(self.parameters()).device)

        x_in = x.view(-1, self.config.num_agents * self.config.num_items)

        # Allocation Network
        a = torch.matmul(x_in, self.w_a[0]) + self.b_a[0]
        a = self.activation(a)
        activation_summary(a)
        
        for i in range(1, self.config.net.num_a_layers - 1):
            a = torch.matmul(a, self.w_a[i]) + self.b_a[i]
            a = self.activation(a)
            activation_summary(a)

        a = torch.matmul(a, self.w_a[-1]) + self.b_a[-1]
        a = F.softmax(a.view(-1, self.config.num_agents + 1, self.config.num_items + 1), dim=1)
        a = a[:, :self.config.num_agents, :self.config.num_items]  # Equivalent to tf.slice
        activation_summary(a)

        # Payment Network
        p = torch.matmul(x_in, self.w_p[0]) + self.b_p[0]
        p = self.activation(p)
        activation_summary(p)

        for i in range(1, self.config.net.num_p_layers - 1):
            p = torch.matmul(p, self.w_p[i]) + self.b_p[i]
            p = self.activation(p)
            activation_summary(p)

        p = torch.matmul(p, self.w_p[-1]) + self.b_p[-1]
        p = torch.sigmoid(p)
        activation_summary(p)
        
        u = torch.sum(a * x.view(-1, self.config.num_agents, self.config.num_items), dim=-1)
        p = p * u
        activation_summary(p)
        
        return a, p

    def compute_allocation_constraint_violation(self, alloc):
        """
        財3の配分確率に対する制約違反を計算（単一bidderの場合）
        
        制約:
        - 下界: max(0, alloc1 + alloc2 - 1) <= alloc3
        - 上界: alloc3 <= min(alloc1, alloc2)
        
        Args:
            alloc: [batch_size, num_agents, num_items]
                  単一bidderの場合、alloc[:, 0, :] が配分確率
        
        Returns:
            constraint_violation: [batch_size]
                                各サンプルごとの制約違反の合計
        """
        assert self.config.num_agents == 1, "制約付き設定ではnum_agents=1である必要があります"
        assert self.config.num_items == 3, "制約付き設定ではnum_items=3である必要があります"
        
        # 単一bidderの場合、alloc[:, 0, :] が配分確率
        alloc1 = alloc[:, 0, 0]  # 財1の配分確率
        alloc2 = alloc[:, 0, 1]  # 財2の配分確率
        alloc3 = alloc[:, 0, 2]  # 財3の配分確率
        
        # 下界: max(0, alloc1 + alloc2 - 1)
        lower_bound = torch.clamp(alloc1 + alloc2 - 1, min=0.0)
        # 上界: min(alloc1, alloc2)
        upper_bound = torch.minimum(alloc1, alloc2)
        
        # 制約違反
        lower_violation = F.relu(lower_bound - alloc3)  # 下界違反（alloc3が小さすぎる）
        upper_violation = F.relu(alloc3 - upper_bound)   # 上界違反（alloc3が大きすぎる）
        
        constraint_violation = lower_violation + upper_violation
        
        return constraint_violation