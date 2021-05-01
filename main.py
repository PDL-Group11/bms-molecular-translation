import utils
from runner import Runner
from model import get_model

import os
import argparse
from glob import glob
import random
import torch
import torch.nn as nn
import torch.cuda as cuda
from pathlib import Path

def arg_parse():
    desc = "BMS Molecular Translation"
    parser = argparse.ArgumentParser(description=desc)

    # System configuration
    parser.add_argument('--gpus', type=str, default="0,1,2,3,4,5,6",
                        help="Select GPUs (Default : Maximum number of available GPUs)")
    parser.add_argument('--cpus', type=int, default="32",
                        help="Select the number of CPUs")

    # Directories
    parser.add_argument('--save_dir', type=str, required=False,
                        help='Directory name to save the model')

    # Model
    parser.add_argument('--model', type=str, default='swin', choices=["swin"], # models might be added 
                        help='model type')

    # Training configuration
    parser.add_argument('--epoch', type=int, default=200, help='epochs')
    parser.add_argument('--batch_train', type=int, default=32, help='size of batch for train')
    parser.add_argument('--batch_test',  type=int, default=32, help='size of batch for tevalidation and test')

    parser.add_argument('--extract', action="store_true", help='feature extraction')
    parser.add_argument('--test', action="store_true", help='test only (skip training)')
    parser.add_argument('--voting', action="store_true", help='majority vote only (skip training)')
    parser.add_argument('--load_last', action="store_true", help='load last saved model')
    parser.add_argument('--load_path', type=str, default=None, help='Absolute path of models to load')
    parser.add_argument('--log_dir', type=str, default=None, help='Path of logs')

    parser.add_argument('--datatype', type=str, help='Path of logs')
    
    # Optimizer
    parser.add_argument('--lr',   type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument('--beta', type=float, default=(0.9, 0.999), nargs="*",
                        help="parameter beta for Adam optimizer")
    return parser.parse_args()


if __name__ == "__main__":

    arg = arg_parse()

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    device = torch.device("cuda")

    arg.save_dir = "/saved_model"
    arg.log_dir = "/log_dir"
    
    os.makedirs(f'/{arg.log_dir}/', exist_ok=True)
    os.makedirs(f'/{arg.save_dir}/', exist_ok=True)

    train_path, val_path, test_path = get_data_path()
    train_loader, val_loader, test_loader = get_loader(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_train=arg.batch_train,
        batch_test=arg.batch_test,
    )

    #TODO: load model
    net = get_model(arg)
    net = nn.DataParallel(net).to(device)
    
    #TODO: add loss criterion
    loss = 
    model = Runner(arg, device, net, train_loader, val_loader, test_loader, loss)

    if arg.load_last:
        print(" === load last trained model ===")
        model.load()
    if arg.load_path:
        print(" === load model from {arg.load_path} ===")
        model.load(abs_filename=arg.load_path)
    if arg.test:
        print(" === inference === ")
        model.test()
    else:
        model.train()
