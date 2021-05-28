import os
import time
import datetime
# from runner import Runner
from models import get_model
from data_loader import get_data, get_loader
import argparse
from glob import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import reference.utils as utils
from reference.engine import train_one_epoch, evaluate

# import torchvision
# import torchvision.utils
# import cv2
# import einops

def arg_parse():
    desc = "BMS Molecular Translation"
    parser = argparse.ArgumentParser(description=desc)

    # System configuration
    parser.add_argument('--cpus', type=int, default="32",
                        help="Select the number of CPUs")
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Directories
    parser.add_argument('--save_dir', type=str,required=False,
                        help='Directory name to save the model')

    # Model
    parser.add_argument('--model', type=str, default="fast_rcnn", choices=["fast_rcnn", "mask_rcnn"],
                        help='model type')
    parser.add_argument('--backbone', type=str, default="resnet_50", choices=["resnet_50", "mobilenet_v2", "swin_transformer"],
                        help='backbone or classifier of detector')
    parser.add_argument('--num_classes', type=int, default=24,  
                        help='number of classes to be detected')

    # Training configuration
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200, help='epochs')
    parser.add_argument('--batch', type=int, default=10, help='size of batch for train')

    parser.add_argument('--test', action="store_true", help='test only (skip training)')
    parser.add_argument('--load_last', action="store_true", help='load last saved model')
    parser.add_argument('--load_path', type=str, default=None, help='Absolute path of models to load')
    parser.add_argument('--log_dir', type=str, default=None, help='Path of logs')

    parser.add_argument('--datatype', type=str, help='Path of logs')
    
    # Optimizer
    parser.add_argument('--lr',   type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')

    parser.add_argument('--lr_scheduler', default='cosineannealinglr', choices=['cosineannealinglr', 'multisteplr'])
    parser.add_argument('--beta', type=float, default=(0.9, 0.999), nargs="*",
                        help="parameter beta for Adam optimizer")
    return parser.parse_args()


if __name__ == "__main__":

    arg = arg_parse()
    arg.save_dir = f"./outs/{arg.backbone}"
    
    if arg.save_dir:
        utils.mkdir(arg.save_dir)
    utils.init_distributed_mode(arg)
    print(arg)

    device = torch.device("cuda")

    root, pkl = get_data()
    train_loader, val_loader, test_loader = get_loader(arg, root, pkl)

    #TODO: get model with backbone (Swin Transformer) and classifier 
    model = get_model(arg, pretrained=True)
    model = model.to(device)
    # model = nn.DataParallel(model).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[arg.gpu])
    model_without_ddp = model.module
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=arg.lr, betas=arg.beta)
    optimizer = torch.optim.SGD(
        params, lr=arg.lr, momentum=arg.momentum, weight_decay=arg.weight_decay)

    arg.lr_scheduler = arg.lr_scheduler.lower()
    if arg.lr_scheduler == 'multisteplr':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.lr_steps, gamma=arg.lr_gamma)
    elif arg.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg.epoch)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(arg.lr_scheduler))

    if arg.load_last:
        print(" === load last trained model ===")
        checkpoint = torch.load(arg.load_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        arg.start_epoch = checkpoint['epoch'] + 1

    # if arg.load_path:
    #     print(" === load model from {arg.load_path} ===")
        # runner.load(abs_filename=arg.load_path)
    if arg.test:
        for i, item in enumerate(test_loader):
            img, label = item
            model = model.load_state_dict(torch.load(arg.load_path)['model'])
            model_without_ddp = model.module

            out = model(img)
            print(out.size())

        # evaluate(model, test_loader, device=device)

    print("Start training")
    early_stopping = utils.EarlyStopping(patience=7, verbose=True)

    start_time = time.time()
    for epoch in range(arg.start_epoch, arg.epoch):
        # train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        lr_scheduler.step()

        # evaluate after every epoch
        loss = evaluate(model, val_loader, device=device)
        early_stopping(arg, epoch, loss, model_without_ddp, optimizer, lr_scheduler)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
