import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


if __name__ == '__main__':
    to_img = ToPILImage()
    train_dataset = datasets.ImageFolder(
        root='./dataset/train/',
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8)
    
    for i, data in enumerate(train_dataloader):
        print('input image size:', data[0].size())
        print('class label:', data[1])
        img = data[0][0, :]
        print('image size: ', img.size())
        print("max: {}, min: {}".format(np.max(img.numpy()), np.min(img.numpy())))
        plt.imshow(to_img(img))
