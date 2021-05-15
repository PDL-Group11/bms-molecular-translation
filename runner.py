import os
from glob import glob
from collections import defaultdict

import torch
from torchvision.de
from reference.engine import train_one_epoch, evaluate
from sklearn.metrics import roc_curve

from .BaseRunner import BaseRunner
from utils import BinaryMetrics, get_optimal_threshold

import numpy as np
import csv

class Runner:
    def __init__(self, arg, device, model, train_loader, val_loader, test_loader):
        super().__init__()
        
        self.arg = arg
        self.device = device

        self.model = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        params = [p for p in model.parameters() if p.requires_grad]
        self.optim = torch.optim.Adam(params, lr=arg.lr, betas=arg.beta)
        self.best_loss = 9999.0
        self.last_filename = ""
        self.save_path = arg.save_dir

        self.log_outs = defaultdict(list)
        self.cluster_outs = []
    
    def save(self, epoch, filename):
        if epoch < 10:
            return
        torch.save({"model_type"  : self.model_type,
                    "start_epoch" : epoch + 1,
                    "network"     : self.model.state_dict(),
                    "optimizer"   : self.optim.state_dict(),
                    "best_loss"   : self.best_loss,
                    }, self.save_path + f"/{filename}.pth.tar")
        print(f"Model saved {epoch} epoch")

    def save_check(self, bm, log_dict, epoch):
        total_loss = log_dict["total_loss"]
        if epoch == 1:
            self.filename = "VAL_epoch[%05d]_checkpoint" % (epoch)
            self.last_filename = self.filename
            self.save(epoch, self.filename)

        if total_loss < self.best_loss:
            self.best_loss = total_loss
            self.filename = "VAL_epoch[%05d]_loss[%f]" % (epoch, total_loss)
            # self.last_filename = self.filename
            self.save(epoch, self.filename)
        
        if epoch % 25 == 0:
            self.filename = "VAL_epoch[%05d]_checkpoint" % (epoch)
            self.save(epoch, self.filename)
        
        if epoch == 199:
            self.filename = "VAL_epoch[%05d]_checkpoint" % (epoch)
            self.save(epoch, self.filename)

    def load(self, filename=None, abs_filename=None):
        os.makedirs(self.save_path, exist_ok=True)
        if filename is None and abs_filename is None:
            # load last epoch model
            filename = sorted(glob(self.save_path + "/*.pth.tar"))[-1]
        elif filename is not None:
            filename = self.save_path + '/' + filename 
        elif abs_filename is not None:
            filename = abs_filename
        self.filename = os.path.basename(filename)
        print(filename)

        if os.path.exists(filename) is True:
            print("Load %s File" % (filename))
            ckpoint = torch.load(filename)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s" % (ckpoint["model_type"]))

            self.model.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch  = ckpoint['start_epoch']
            self.best_loss    = ckpoint["best_loss"]
            print("Load Model Type : %s, last epoch : %d, best_loss : %f" % (
                  ckpoint["model_type"], self.start_epoch - 1, self.best_loss))
        else:
            print("Load Failed, file not exists")

    def train(self):
        print("===== Training ===== \n")
        # self.model.train()
        for epoch in range(self.start_epoch, self.epoch):
            log_dict = defaultdict(int)
            metric_logger = train_one_epoch(self.model, self.optim, self.train_loader, self.device, epoch, 10)
            print(metric_logger)
            # for i, items in enumerate(self.train_loader):
            #     self._step_train(items, log_dict)
            #     if i % 10 == 0:
            #         print("Training - %d Epoch [%d / %d]" % (epoch, i, len(self.train_loader)))
            # self._log_train(epoch, log_dict)
            self.valid(epoch)
            self.save_logs()
        self.test()
    
    def _step_train(self, items, log_dict):
        input_, labels = items
        
        labels = labels.to(self.device, dtype=torch.float32)

        regressed = self.model(input_, labels)
        loss = self.loss(regressed, labels)
        log_dict["total_loss"] += (loss.item() / len(self.train_loader.dataset))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
    def _log_train(self, epoch, log_dict):
        total_loss = log_dict["total_loss"]
        self.log_outs['train_loss'].append([epoch, total_loss])
    
    def valid(self, epoch, return_bm=False):
        print("===== Validation after epoch = {} =====\n".format(epoch))
        bm = BinaryMetrics()
        # self.model.eval()
        # with torch.no_grad():
        log_dict = defaultdict(int)
        # for i, items in enumerate(self.val_loader):
        #     self._step_valid(items, bm, log_dict)
        evaluator = evaluate(self.model, self.val_loader, self.device)
        print(evaluator)
        # if return_bm:
        #     return bm

        self._log_valid(epoch, bm, log_dict)
        self.save_check(bm, log_dict, epoch)

    def _step_valid(self, items, bm, log_dict):
        input_, labels = items
        labels = labels.to(self.device, dtype=torch.float32)

        regressed = self.model(input_)
        loss = self.loss(regressed, labels)

        regressed = regressed.cpu().numpy()
        labels = labels.cpu().numpy()

        log_dict["total_loss"] += (loss.item() / len(self.val_loader.dataset))

        bm(regressed, labels)

    def _log_valid(self, epoch, bm, log_dict):
        bm.calc_metric()
        total_loss = log_dict["total_loss"]
        self.log_outs['train_bm'].append([epoch, bm.matric_dict])
        print("Valid accuracy: {:.3%}".format(bm.matric_dict['accuracy']))
    
    def test(self):
        def _get_val_opt_threshold():
            val_bm = self.valid(self.start_epoch - 1, return_bm=True)
            roc_curve_arrays = roc_curve(val_bm.labels, val_bm.regressed, pos_label=0)
            val_opt_threshold = get_optimal_threshold(roc_curve_arrays)
            return val_opt_threshold
        
        # bm = BinaryMetrics()

        print(" ===== Testing ===== \n")
        # self.load(filename=self.last_filename + ".pth.tar")
        self.load(abs_filename=self.arg.load_path)
        # self.model.eval()
        # with torch.no_grad():
        evaluator = evaluate(self.model, self.test_loader, self.device)
            # log_dict = defaultdict(int)
            # for i, items in enumerate(self.test_loader):
            #     print(f"----------{i}----------")
            #     target = items[2:]
            #     self._step_test(target, bm, log_dict)
            #     bm.calc_metric()
            #     items.pop(2)
            #     items[0] = items[0][0]
            #     items[1] = items[1].item()
            #     items[2] = items[2].item()
            #     items.append(bm.regressed[-1])
            #     self.cluster_outs.append(items)
        print(evaluator)
        self.save_logs()

        print(" ===== Finish Testing ===== \n")

    def _step_test(self, target, bm, log_dict):
        input_, labels = target
        labels = labels.to(self.device, dtype=torch.float32)

        regressed = self.model(input_)
        loss = self.loss(regressed, labels)

        regressed = regressed.cpu().numpy()
        labels = labels.cpu().numpy()
        
        bm(regressed, labels)
    
    def save_logs(self):
        log_csv = open(f"/{self.arg.log_dir}_log.csv", 'w')
        writer = csv.writer(log_csv)
        writer.writerows(self.log_outs)
        log_csv.close()

        out_csv = open(f"/{self.arg.log_dir}_output.csv", 'w')
        writer = csv.writer(out_csv)
        writer.writerows(self.cluster_outs)
        out_csv.close()
