from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from base.base_generator import BaseGenerator


class Generator(BaseGenerator):
    """
    3財のデータジェネレーター
    - 財1, 財2: uniform [0, 1]
    - 財3: uniform [0, c] (cはパラメータ)
    """
    
    def __init__(self, config, mode, X=None, ADV=None):
        super(Generator, self).__init__(config, mode)
        # cパラメータをconfigから取得（デフォルト1.0）
        self.c = getattr(config, 'c', 1.0)
        self.build_generator(X=X, ADV=ADV)
    
    def generate_random_X(self, shape):
        """
        財1,2は[0,1]のuniform、財3は[0,c]のuniform
        shape: [batch_size, num_agents, num_items] (num_items=3)
        """
        X = np.random.rand(*shape)
        # 財3（index 2）を[0,c]にスケール
        X[:, :, 2] = X[:, :, 2] * self.c
        return X
    
    def generate_random_ADV(self, shape):
        """
        財1,2は[0,1]のuniform、財3は[0,c]のuniform
        shape: [num_misreports, batch_size, num_agents, num_items]
        """
        ADV = np.random.rand(*shape)
        # 財3（index 2）を[0,c]にスケール
        ADV[:, :, :, 2] = ADV[:, :, :, 2] * self.c
        return ADV

