import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'

from runner import Runner
from models import get_model
from data_loader import get_data, get_loader
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
    parser.add_argument('--gpus', type=str, default="0,1,2",
                        help="Select GPUs (Default : Maximum number of available GPUs)")
    parser.add_argument('--cpus', type=int, default="32",
                        help="Select the number of CPUs")

    # Directories
    parser.add_argument('--save_dir', type=str, required=False,
                        help='Directory name to save the model')

    # Model
    parser.add_argument('--model', type=str, default="fast_rcnn", choices=["fast_rcnn", "mask_rcnn"],
                        help='model type')
    parser.add_argument('--backbone', type=str, default="mobilenet_v2", choices=["resnet_50", "mobilenet_v2", "swin_transformer"],
                        help='backbone or classifier of detector')
    parser.add_argument('--num_classes', type=int, default=35,  
                        help='number of classes to be detected')

    # Training configuration
    parser.add_argument('--epoch', type=int, default=200, help='epochs')
    parser.add_argument('--batch_train', type=int, default=30, help='size of batch for train')
    parser.add_argument('--batch_test',  type=int, default=30, help='size of batch for tevalidation and test')

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

    #os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    device = torch.device("cuda")

    arg.save_dir = f"faster_rcnn/model_{arg.backbone}"
    arg.log_dir = f"faster_rcnn/log_{arg.backbone}"
    
    os.makedirs(f'./outs/{arg.log_dir}/', exist_ok=True)
    os.makedirs(f'./outs/{arg.save_dir}/', exist_ok=True)

    root, pkl, transform = get_data()
    train_loader, val_loader, test_loader = get_loader(arg, root, pkl, transform)

    #TODO: get model with backbone (Swin Transformer) and classifier 
    model = get_model(arg, pretrained=True)
    model = nn.DataParallel(model).to(device)
    
    #TODO: add loss criterion
    runner = Runner(arg, device, model, train_loader, val_loader, test_loader)

    if arg.load_last:
        print(" === load last trained model ===")
        runner.load()
    if arg.load_path:
        print(" === load model from {arg.load_path} ===")
        runner.load(abs_filename=arg.load_path)
    if arg.test:
        print(" === inference === ")
        runner.test()
    else:
        runner.train()
