from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from base.base_generator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, X = None):
        super(Generator, self).__init__(config, mode)
        # 相関0.95の共分散行列
        self.cov = np.array([[1.0, 0.95], [0.95, 1.0]])
        self.mean = np.array([0.0, 0.0])
        self.build_generator(X = X)

    def generate_random_X(self, shape):
        # 多変量正規分布からサンプリング
        samples = np.random.multivariate_normal(self.mean, self.cov, size=shape[0])
        # 標準正規分布を[0,1]の範囲に変換（平均0.5、標準偏差0.2程度にスケール）
        # 3シグマルールを使用して、ほぼ全ての値が[0,1]の範囲に入るようにする
        samples = samples * 0.2 + 0.5
        # [0,1]の範囲にクリップ
        samples = np.clip(samples, 0.0, 1.0)
        return samples

