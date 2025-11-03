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


class Trainer(object):

    def __init__(self, config, mode, net):
        self.config = config
        self.mode = mode
        
        # Create output-dir
        if not os.path.exists(self.config.dir_name): os.mkdir(self.config.dir_name)

        if self.mode == "train":
            log_suffix = '_' + str(self.config.train.restore_iter) if self.config.train.restore_iter > 0 else ''
            self.log_fname = os.path.join(self.config.dir_name, 'train' + log_suffix + '.txt')
        else:
            log_suffix = "_iter_" + str(self.config.test.restore_iter)
            self.log_fname = os.path.join(self.config.dir_name, "test" + log_suffix + ".txt")
            
        # Set Seeds for reproducibility
        np.random.seed(self.config[self.mode].seed)
        torch.manual_seed(self.config[self.mode].seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config[self.mode].seed)
        
        # Init Logger
        self.init_logger()

        # Init Net
        self.net = net
        
        # Init graph components (no TF graph, just setup optimizers)
        self.init_graph()
              
    def init_logger(self):

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.FileHandler(self.log_fname, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.logger = logger

    def compute_rev(self, pay):
        """ Given payment (pay), computes revenue
            Input params:
                pay: [num_batches]
            Output params:
                revenue: scalar
        """
        return torch.mean(pay)
    
    def compute_utility(self, x, alloc, pay):
        """ Given input valuation (x), payment (pay) and allocation (alloc), computes utility
            Input params:
                x: [num_batches, num_items]
                a: [num_batches, num_items]
                p: [num_batches]
            Output params:
                utility: [num_batches]
            """
        return torch.sum(alloc * x, dim=-1) - pay


    def init_graph(self):
        # Weight decay parameter
        wd = None if "wd" not in self.config.train else self.config.train.wd
        
        # Optimizer
        learning_rate = self.config.train.learning_rate
        self.opt = optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=wd if wd else 0)

        # Metrics names
        self.metric_names = ["Net_Loss", "Revenue"]

    def compute_metrics(self, x):
        """Compute metrics given input x"""
        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x

        # Get mechanism for true valuation
        alloc, pay = self.net.inference(x_tensor)
        
        # Metrics
        revenue = self.compute_rev(pay)
        utility = self.compute_utility(x_tensor, alloc, pay) 
        irp_mean = torch.mean(F.relu(-utility))

        loss = -revenue
        
        metrics = [loss.item(), revenue.item()]
        return metrics, alloc, pay
        
    def train(self, generator):
        """
        Runs training
        """
        
        self.train_gen, self.val_gen = generator
        
        iter = self.config.train.restore_iter

        if iter > 0:
            model_path = os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt')
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.opt.load_state_dict(checkpoint['opt_state_dict'])

        if iter == 0:
            self.train_gen.save_data()
            torch.save({
                'net_state_dict': self.net.state_dict(),
                'opt_state_dict': self.opt.state_dict(),
                'iter': iter
            }, os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt'))

        time_elapsed = 0.0       
        while iter < (self.config.train.max_iter):
             
            tic = time.time()
                
            # Get a mini-batch
            X = next(self.train_gen.gen_func)
            X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
            
            self.net.train()
            self.opt.zero_grad()
            alloc, pay = self.net.inference(X_tensor)
            revenue = self.compute_rev(pay)
            loss = -revenue
            loss.backward()
            self.opt.step()
            
            # Clip alpha to [0, 1] (commented out in original, but keeping for compatibility)
            # self.net.alpha.data.clamp_(min=0.0, max=1.0)
                
            iter += 1

            toc = time.time()
            time_elapsed += (toc - tic)
                        
            if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter): 
                torch.save({
                    'net_state_dict': self.net.state_dict(),
                    'opt_state_dict': self.opt.state_dict(),
                    'iter': iter
                }, os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt'))

            if (iter % self.config.train.print_iter) == 0:
                # Train Set Stats
                metrics, _, _ = self.compute_metrics(X_tensor.detach())
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
                        X_tensor = torch.tensor(X, dtype=torch.float32)
                        metrics, _, _ = self.compute_metrics(X_tensor)
                        metric_tot += metrics
                    
                    metric_tot = metric_tot / self.config.val.num_batches
                    fmt_vals = tuple([item for tup in zip(self.metric_names, metric_tot) for item in tup])
                    log_str = "VAL-%d" % (iter) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
                    self.logger.info(log_str)

    def test(self, generator):
        """
        Runs test
        """
        
        # Init generators
        self.test_gen = generator

        iter = self.config.test.restore_iter

        model_path = os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt')
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['net_state_dict'])

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
                X_tensor = torch.tensor(X, dtype=torch.float32)
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
