import torchvision.datasets as datasets
from torchvision.transforms.transforms import Resize
#import torchvision.transforms as transforms
import reference.transforms as transforms
import torch
import torchvision
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, functional
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import os
import pickle
import cv2
import einops
from reference.utils import is_use_distributed_mode

from reference.utils import collate_fn

class TestDataset(object):

    def __init__(self, root, csv, transform=None):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """

        self.data = datasets.ImageFolder(root)
        self.label_frame = pd.read_csv(csv)
        self.filename = self.label_frame['image_id']
        self.transform = transform

    def __len__(self):
        """return number of points in our dataset"""

        return len(self.label_frame)

    def get_image_from_folder(self, filename):
        file_path = [filename[0], filename[1], filename[2], f"{filename}.png"]
        f = os.path.join(self.data.root, *file_path)
        img = self.data.loader(f)
        return img

    def __getitem__(self, idx):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """
        if torch.is_tensor(idx):
            idx = idx.item()

        img = self.get_image_from_folder(self.filename[idx])

        if self.transform:
            img = self.transform(img)
        return img, self.filename[idx]

class DetectionDataset(object):

    def __init__(self, root, pkl, transform=None):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """

        self.data = datasets.ImageFolder(root, transform)
        with open(pkl, 'rb') as f:
            label = pickle.load(f)
        self.label = label
        self.transform = transform

    def __len__(self):
        """return number of points in our dataset"""

        return len(self.label)

    def __getitem__(self, idx):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """
        img = self.data[idx][0]
        _label = self.label[idx]
        image_id = _label['image_id']
        boxes = []
        labels = []
        objs = _label['annotations']

        for i in range(len(objs)):
            obj = objs[i]
            box = obj['bbox']
            bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            boxes.append(bbox)
            labels.append(obj['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objs),), dtype=torch.int64)

        label = {}
        label['boxes'] = boxes
        label['labels'] = labels
        label['image_id'] = image_id
        label['area'] = area
        label['iscrowd'] = iscrowd

        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label

class MoleculeDetectionDataset(DetectionDataset):

    def __init__(self, root, pkl, transform=None):
        '''
        args:
            pkl (string): label pkl path
            root (string): image files path
            transform (callable, optional): optional transform on samples
        '''
        super(MoleculeDetectionDataset, self).__init__(root, pkl)
        self.root = root
        self.transform = transform

def get_data():
    
    root = {
        # 'train': './dataset/good2/train/',
        'train': './dataset/all_detection/train/',
        # 'val': './dataset/good2/val/',
        'val': './dataset/all_detection/val/',
        # 'test': './dataset/test/'
        'test': '~/hgkim/bms-molecular-translation/dataset/new_test/'
    }

    pkl = {
        # 'train': './dataset/good2/train_annotations_train.pkl',
        'train': './dataset/all_annotations_train.pkl',
        # 'val': './dataset/good2/train_annotations_val.pkl',
        'val': './dataset/all_annotations_val.pkl',
        'test': './dataset/sample_submission.csv'
    }
    return root, pkl


def get_loader(arg, root, pkl):

    transform = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize((300, 300)),
            torchvision.transforms.ToTensor()
        ])
    }

    train_dataset = MoleculeDetectionDataset(root['train'], pkl['train'], transform['train'])
    val_dataset = MoleculeDetectionDataset(root['val'], pkl['val'], transform['val'])
    # test_dataset = MoleculeDetectionDataset(root['test'], pkl['train'], transform['test'])
    test_dataset = TestDataset(root['test'], pkl['test'], transform['test'])

    if is_use_distributed_mode(arg):
        train_samlper = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        test_sampler = DistributedSampler(test_dataset)
    
        train_batch_sampler = torch.utils.data.BatchSampler(train_samlper, arg.batch, drop_last=True)
        test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, 1, drop_last=False)

        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=8, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=arg.batch, sampler=val_sampler, num_workers=8, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, batch_sampler=test_batch_sampler, shuffle=False, num_workers=8)
    else:
        train_loader = DataLoader(train_dataset, num_workers=8, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=arg.batch, num_workers=8, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    # train_dataset = MoleculeDetectionDataset(
    #     root='./dataset/all_detection/train',
    #     pkl='./dataset/all_annotations_train.pkl',
    #     transform=transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    # )

    # val_dataset = MoleculeDetectionDataset(
    #     root='./dataset/all_detection/val',
    #     pkl='./dataset/all_annotations_val.pkl',
    #     transform=transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    # )

    # train_dataloader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     collate_fn = collate_fn
    # )
    
    transform = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ])
    }
    test_dataset = TestDataset('./dataset/test/', './dataset/sample_submission.csv', torchvision.transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, item in enumerate(test_loader):
        if i == 0:
            img, filename = item
            print(img.size(), filename)
        else:
            break