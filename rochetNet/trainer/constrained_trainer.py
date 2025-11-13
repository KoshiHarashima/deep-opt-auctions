from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from trainer.trainer import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    """
    BaseTrainerを継承し、配分確率制約違反の罰則項を追加（Augmented Lagrangian法）
    
    財3の配分確率に対する制約:
    - 下界: max(0, alloc1 + alloc2 - 1) <= alloc3
    - 上界: alloc3 <= min(alloc1, alloc2)
    
    最小化する関数:
    L = -revenue + λ_constraint · constraint_violation + (ρ_constraint/2) · constraint_violation²
    """
    
    def __init__(self, config, mode, net):
        super(Trainer, self).__init__(config, mode, net)
    
    def compute_allocation_constraint_violation(self, alloc):
        """
        ネットワークの制約違反計算関数を呼び出し
        """
        if hasattr(self.net, 'compute_allocation_constraint_violation'):
            return self.net.compute_allocation_constraint_violation(alloc)
        else:
            # フォールバック（通常は発生しない）
            return torch.zeros(alloc.shape[0], device=alloc.device)
    
    def init_graph(self):
        """
        BaseTrainerのinit_graphを呼び出し、制約違反用のパラメータを追加
        """
        super(Trainer, self).init_graph()
        
        # 制約違反用のLagrange乗数（train/test両方で必要）
        w_constraint_init_val = 5.0
        if self.mode == "train" and "w_constraint_init_val" in self.config.train:
            w_constraint_init_val = self.config.train.w_constraint_init_val
        self.w_constraint = nn.Parameter(
            torch.tensor(w_constraint_init_val, dtype=torch.float32, device=self.device)
        )
        
        # 制約違反用の更新レート（ペナルティパラメータ）
        constraint_update_rate = 1.0
        if self.mode == "train" and "constraint_update_rate" in self.config.train:
            constraint_update_rate = self.config.train.constraint_update_rate
        self.constraint_update_rate = nn.Parameter(
            torch.tensor(constraint_update_rate, dtype=torch.float32, device=self.device),
            requires_grad=False
        )
        
        if self.mode == "train":
            # 制約違反用のオプティマイザー（Lagrange乗数更新用）
            self.opt_constraint = optim.SGD([self.w_constraint], lr=self.constraint_update_rate.item())
            
            # ペナルティパラメータの増加量
            self.constraint_update_rate_add = 0.1 if "constraint_update_rate_add" not in self.config.train else self.config.train.constraint_update_rate_add
            
            # 更新頻度
            self.update_frequency = 100 if "update_frequency" not in self.config.train else self.config.train.update_frequency
            self.up_op_frequency = 1000 if "up_op_frequency" not in self.config.train else self.config.train.up_op_frequency
        
        # メトリクス名に追加
        self.metric_names = [
            "Net_Loss", "Revenue",
            "Constraint_Viol", "Const_Penalty", "Const_Lag_Loss", "w_constraint"
        ]
    
    def train(self, generator):
        """
        BaseTrainerのtrainをオーバーライドし、制約違反の処理を追加
        """
        self.train_gen, self.val_gen = generator
        
        iter = self.config.train.restore_iter
        
        if iter > 0:
            model_path = os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt')
            checkpoint = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.opt.load_state_dict(checkpoint['opt_state_dict'])
            if 'w_constraint' in checkpoint:
                self.w_constraint.data = checkpoint['w_constraint'].to(self.device)
            if 'constraint_update_rate' in checkpoint:
                self.constraint_update_rate.data = checkpoint['constraint_update_rate'].to(self.device)
            if 'opt_constraint_state_dict' in checkpoint:
                self.opt_constraint.load_state_dict(checkpoint['opt_constraint_state_dict'])
        
        if iter == 0:
            self.train_gen.save_data()
            save_dict = {
                'net_state_dict': self.net.state_dict(),
                'opt_state_dict': self.opt.state_dict(),
                'w_constraint': self.w_constraint.data,
                'constraint_update_rate': self.constraint_update_rate.data,
                'opt_constraint_state_dict': self.opt_constraint.state_dict(),
                'iter': iter
            }
            torch.save(save_dict, os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt'))
        
        time_elapsed = 0.0
        while iter < (self.config.train.max_iter):
            
            tic = time.time()
            
            # Get a mini-batch
            X = next(self.train_gen.gen_func)
            X_tensor = torch.from_numpy(X).to(self.device).float().requires_grad_(True)
            
            if iter == 0:
                # Initial Lagrange update (constraint)
                self.net.eval()
                alloc, pay = self.net.inference(X_tensor)
                constraint_violation = self.compute_allocation_constraint_violation(alloc)
                constraint_violation_mean = torch.mean(constraint_violation)
                constraint_lag_loss = -self.w_constraint * constraint_violation_mean
                self.opt_constraint.zero_grad()
                constraint_lag_loss.backward()
                self.opt_constraint.step()
            
            self.net.train()
            self.opt.zero_grad()
            alloc, pay = self.net.inference(X_tensor)
            revenue = self.compute_rev(pay)
            
            # Constraint violation computation
            constraint_violation = self.compute_allocation_constraint_violation(alloc)
            constraint_violation_mean = torch.mean(constraint_violation)
            
            # Augmented Lagrangian loss
            constraint_penalty = self.constraint_update_rate * constraint_violation_mean ** 2 / 2.0
            constraint_lag_loss = self.w_constraint * constraint_violation_mean
            loss = -revenue + constraint_penalty + constraint_lag_loss
            
            loss.backward()
            self.opt.step()
            
            iter += 1
            
            # Run Lagrange Update
            if iter % self.update_frequency == 0:
                self.net.eval()
                alloc, pay = self.net.inference(X_tensor)
                constraint_violation = self.compute_allocation_constraint_violation(alloc)
                constraint_violation_mean = torch.mean(constraint_violation)
                loss_constraint = -self.w_constraint * constraint_violation_mean
                self.opt_constraint.zero_grad()
                loss_constraint.backward()
                self.opt_constraint.step()
            
            # Update penalty parameter
            if iter % self.up_op_frequency == 0:
                self.constraint_update_rate.data += self.constraint_update_rate_add
                # Update optimizer learning rate
                for param_group in self.opt_constraint.param_groups:
                    param_group['lr'] = self.constraint_update_rate.item()
            
            toc = time.time()
            time_elapsed += (toc - tic)
            
            if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter):
                save_dict = {
                    'net_state_dict': self.net.state_dict(),
                    'opt_state_dict': self.opt.state_dict(),
                    'w_constraint': self.w_constraint.data,
                    'constraint_update_rate': self.constraint_update_rate.data,
                    'opt_constraint_state_dict': self.opt_constraint.state_dict(),
                    'iter': iter
                }
                torch.save(save_dict, os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt'))
            
            if (iter % self.config.train.print_iter) == 0:
                # Train Set Stats
                self.net.eval()
                with torch.no_grad():
                    alloc, pay = self.net.inference(X_tensor)
                    revenue = self.compute_rev(pay)
                    constraint_violation = self.compute_allocation_constraint_violation(alloc)
                    constraint_violation_mean = torch.mean(constraint_violation)
                    constraint_penalty = self.constraint_update_rate * constraint_violation_mean ** 2 / 2.0
                    constraint_lag_loss = self.w_constraint * constraint_violation_mean
                    loss = -revenue + constraint_penalty + constraint_lag_loss
                    
                    metrics = [
                        loss.item(), revenue.item(),
                        constraint_violation_mean.item(), constraint_penalty.item(), 
                        constraint_lag_loss.item(), self.w_constraint.item()
                    ]
                
                fmt_vals = tuple([item for tup in zip(self.metric_names, metrics) for item in tup])
                log_str = "TRAIN-BATCH Iter: %d, t = %.4f" % (iter, time_elapsed) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
                self.logger.info(log_str)
            
            if (iter % self.config.val.print_iter) == 0:
                # Validation Set Stats
                metric_tot = np.zeros(len(self.metric_names))
                self.net.eval()
                with torch.no_grad():
                    for _ in range(self.config.val.num_batches):
                        X = next(self.val_gen.gen_func)
                        X_tensor = torch.from_numpy(X).to(self.device).float()
                        alloc, pay = self.net.inference(X_tensor)
                        revenue = self.compute_rev(pay)
                        constraint_violation = self.compute_allocation_constraint_violation(alloc)
                        constraint_violation_mean = torch.mean(constraint_violation)
                        constraint_penalty = self.constraint_update_rate * constraint_violation_mean ** 2 / 2.0
                        constraint_lag_loss = self.w_constraint * constraint_violation_mean
                        loss = -revenue + constraint_penalty + constraint_lag_loss
                        
                        metrics = [
                            loss.item(), revenue.item(),
                            constraint_violation_mean.item(), constraint_penalty.item(), 
                            constraint_lag_loss.item(), self.w_constraint.item()
                        ]
                        metric_tot += metrics
                    
                    metric_tot = metric_tot / self.config.val.num_batches
                    fmt_vals = tuple([item for tup in zip(self.metric_names, metric_tot) for item in tup])
                    log_str = "VAL-%d" % (iter) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
                    self.logger.info(log_str)
    
    def compute_metrics(self, x):
        """Compute metrics given input x"""
        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).to(self.device).float()
        else:
            x_tensor = x.to(self.device) if x.device != self.device else x

        # Get mechanism for true valuation
        alloc, pay = self.net.inference(x_tensor)
        
        # Metrics
        revenue = self.compute_rev(pay)
        constraint_violation = self.compute_allocation_constraint_violation(alloc)
        constraint_violation_mean = torch.mean(constraint_violation)
        constraint_penalty = self.constraint_update_rate * constraint_violation_mean ** 2 / 2.0
        constraint_lag_loss = self.w_constraint * constraint_violation_mean
        loss = -revenue + constraint_penalty + constraint_lag_loss
        
        metrics = [
            loss.item(), revenue.item(),
            constraint_violation_mean.item(), constraint_penalty.item(), 
            constraint_lag_loss.item(), self.w_constraint.item()
        ]
        return metrics, alloc, pay
    
    def test(self, generator):
        """
        BaseTrainerのtestをオーバーライドし、制約関連パラメータのロードを追加
        """
        # Init generators
        self.test_gen = generator

        iter = self.config.test.restore_iter

        model_path = os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        
        # 制約関連パラメータをロード
        if 'w_constraint' in checkpoint:
            self.w_constraint.data = checkpoint['w_constraint'].to(self.device)
        if 'constraint_update_rate' in checkpoint:
            self.constraint_update_rate.data = checkpoint['constraint_update_rate'].to(self.device)

        # Test-set Stats
        time_elapsed = 0          
        metric_tot = np.zeros(len(self.metric_names))

        if self.config.test.save_output:
            assert(hasattr(generator, "X")), "save_output option only allowed when config.test.data = Fixed or when X is passed as an argument to the generator"
            alloc_tst = np.zeros(self.test_gen.X.shape)
            pay_tst = np.zeros(self.test_gen.X.shape[:-1])

        self.net.eval()
        with torch.no_grad():
            for i in range(self.config.test.num_batches):
                tic = time.time()
                X = next(self.test_gen.gen_func)
                X_tensor = torch.from_numpy(X).to(self.device).float()
                metrics, alloc, pay = self.compute_metrics(X_tensor)
                
                if self.config.test.save_output:
                    alloc_np = alloc.detach().cpu().numpy()
                    pay_np = pay.detach().cpu().numpy()
                    perm = range(i * alloc_np.shape[0], (i + 1) * alloc_np.shape[0])
                    alloc_tst[perm, :] = alloc_np
                    pay_tst[perm] = pay_np
                        
                metric_tot += metrics
                toc = time.time()
                time_elapsed += (toc - tic)
        
        metric_tot = metric_tot / self.config.test.num_batches
        fmt_vals = tuple([item for tup in zip(self.metric_names, metric_tot) for item in tup])
        log_str = "TEST ALL-%d: t = %.4f" % (iter, time_elapsed) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
        self.logger.info(log_str)
        
        if self.config.test.save_output:
            np.save(os.path.join(self.config.dir_name, 'alloc_tst_' + str(iter)), alloc_tst)
            np.save(os.path.join(self.config.dir_name, 'pay_tst_' + str(iter)), pay_tst)

