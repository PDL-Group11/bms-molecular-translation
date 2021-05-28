import os
from glob import glob
from collections import defaultdict

import torch
from reference.engine import train_one_epoch, evaluate
from sklearn.metrics import roc_curve

# from utils import BinaryMetrics, get_optimal_threshold

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
        self.best_iou = 0.0
        self.last_filename = ""
        self.save_path = arg.save_dir
        self.start_epoch = 0
        self.epoch = arg.epoch
        self.log_outs = defaultdict(list)
        self.cluster_outs = []
    
    def save(self, epoch, filename):
        if epoch < 10:
            return
        torch.save({"model_type"  : self.model_type,
                    "start_epoch" : epoch + 1,
                    "network"     : self.model.state_dict(),
                    "optimizer"   : self.optim.state_dict(),
                    "best_iou"   : self.best_iou,
                    }, self.save_path + f"/{filename}.pth.tar")
        print(f"Model saved {epoch} epoch")

    def save_check(self, val_iou, epoch):
        if epoch == 1:
            self.filename = "VAL_epoch[%05d]_checkpoint" % (epoch)
            self.last_filename = self.filename
            self.save(epoch, self.filename)

        if val_iou >= self.best_iou:
            self.best_iou = val_iou
            self.filename = "VAL_epoch[%05d]_iou[%f]" % (epoch, val_iou)
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
            self.best_iou    = ckpoint["best_iou"]
            print("Load Model Type : %s, last epoch : %d, best_iou : %f" % (
                  ckpoint["model_type"], self.start_epoch - 1, self.best_iou))
        else:
            print("Load Failed, file not exists")

    def train(self):
        print(" ===== Training ===== \n")
        for epoch in range(self.start_epoch, self.epoch):
            metric_logger = train_one_epoch(self.model, self.optim, self.train_loader, self.device, epoch, 10)
            self.save_logs(metric_logger)
            self.valid(epoch)
        self.test()
            
    def valid(self, epoch):
        print(" ===== Validation after epoch = {} ===== \n".format(epoch))
        evaluator = evaluate(self.model, self.val_loader, self.device)
        for _, coco_eval in evaluator.coco_eval.items():
            AP = coco_eval.stats
        self.save_logs(evaluator.coco_eval)
        self.save_check(AP[0], epoch)
    
    def test(self):        
        print(" ===== Testing ===== \n")
        self.load(abs_filename=self.arg.load_path)
        evaluator = evaluate(self.model, self.test_loader, self.device, phase='test')
        print(evaluator)
        self.save_logs(evaluator)

        print(" ===== Finish Testing ===== \n")
    
    def save_logs(self, logger):
        file_name = f"./outs/{self.arg.log_dir}/log.txt"

        if os.path.exists(file_name):
            append_write = 'a'
        else:
            append_write = 'w'

        log_file = open(file_name, append_write)
        log_file.write(f"{str(logger)} \n")
        log_file.close()