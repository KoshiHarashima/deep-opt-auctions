from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from base.base_generator import BaseGenerator


class Generator(BaseGenerator):
    """
    3財のデータジェネレーター
    - 財1, 財2: uniform [0, 1]
    - 財3: uniform [0, c] (cはパラメータ、デフォルト1.0)
    
    rochetNetは単一bidderなので、shapeは[batch_size, num_items]
    """
    
    def __init__(self, config, mode, X=None):
        super(Generator, self).__init__(config, mode)
        # cパラメータをconfigから取得（デフォルト1.0）
        self.c = getattr(config, 'c', 1.0)
        assert self.config.num_items == 3, "制約付き設定ではnum_items=3である必要があります"
        self.build_generator(X=X)
    
    def generate_random_X(self, shape):
        """
        財1,2は[0,1]のuniform、財3は[0,c]のuniform
        shape: [batch_size, num_items] (num_items=3)
        """
        X = np.random.rand(*shape)
        # 財3（index 2）を[0,c]にスケール
        X[:, 2] = X[:, 2] * self.c
        return X

