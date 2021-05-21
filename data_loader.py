import torchvision.datasets as datasets
import torchvision.transforms as transforms
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
from reference.utils import collate_fn

class Dataset:

    def __init__(self, root, csv, transform=None):
        """Init function should not do any heavy lifting, but
            must initialize how many items are available in this data set.
        """

        self.data = datasets.ImageFolder(root, transform)
        self.label_frame = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        """return number of points in our dataset"""

        return len(self.label_frame)

    def __getitem__(self, idx):
        """ Here we have to return the item requested by `idx`
            The PyTorch DataLoader class will use this method to make an iterable for
            our training or validation loop.
        """

        img = self.data[idx][0]
        label = self.label_frame.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        return img, label

class DetectionDataset:

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

        label = {}
        label['boxes'] = boxes
        label['labels'] = labels
        label['image_id'] = image_id
        label['area'] = area
        
        if self.transform:
            img = self.transform(img)

        return img, label


class MoleculeDataset(Dataset):

    def __init__(self, root, csv, transform=None):
        '''
        args:
            csv (string): label csv path
            root (string): image files path
            transform (callable, optional): optional transform on samples
        '''
        super(MoleculeDataset, self).__init__(root, csv)
        self.root = root
        self.transform = transform

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
        'train': './dataset/train/',
        'test': './dataset/test/'
    }

    pkl = {
        'train': './dataset/train_annotations_train.pkl',
        'val': './dataset/train_annotations_val.pkl',
        'test': './dataset/train_annotations_val.pkl'
    }

    transform = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.ToTensor()
        ])
    }

    return root, pkl, transform

#def get_loader(arg, root, csv, transform):
#    
#    train_dataset = MoleculeDataset(root['train'], csv['train'], transform['train'])
#    val_dataset = MoleculeDataset(root['train'], csv['train'], transform['val'])
#    test_dataset = MoleculeDataset(root['test'], csv['test'], transform['test'])
#
#    train_dataset = MoleculeDetectionDataset(root['train'], pkl['train'], transform['train'])
#    val_dataset = MoleculeDetectionDataset(root['train'], pkl['val'], transform['val'])
#    test_dataset = MoleculeDetectionDataset(root['test'], pkl['test'], transform['test'])
#
#    train_loader = DataLoader(train_dataset, arg.batch_train, shuffle=True, num_workers=8, collate_fn=collate_fn)
#    val_loader = DataLoader(val_dataset, arg.batch_test, shuffle=True, num_workers=8, collate_fn=collate_fn)
#    test_loader = DataLoader(test_dataset, arg.batch_test, shuffle=False, num_workers=8)
#    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    train_dataset = MoleculeDetectionDataset(
        root='./dataset/train_detection/',
        pkl='./dataset/train_annotations_train.pkl',
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    for i, data in enumerate(train_dataloader):
        print('input image size:', data[0].size())
        print('class label:', data[1])
        print('data[0]:', data[0])
        img = data[0].squeeze(0)#[:]
        print('image size: ', img.size())
        img = img.permute(1, 2, 0)
        print('img:', img)
        print('type(img):', type(img))
        print("max: {}, min: {}, mean: {}".format(np.max(img.cpu().numpy()), np.min(img.cpu().numpy()), np.average(img.cpu().numpy())))
        plt.imshow(functional.to_pil_image(img))
        #plt.imshow(transforms.ToPILImage(np.array(img.squeeze(0))))
        #plt.show()
        