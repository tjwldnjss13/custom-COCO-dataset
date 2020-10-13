import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt


class simpleDataset(data.Dataset):
    def __init__(self, root, filenames, labels):
        self.root = root
        self.filenames = filenames
        self.labels = labels

    def __getitem__(self, index):
        image_fn = self.filenames[index]
        image = Image.open(os.path.join(self.root, image_fn))
        label = self.labels[index]

        image = transforms.ToTensor()(image)
        label = torch.as_tensor(label, dtype=torch.int64)

        return image, label

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    root = 'C://Users/NRS/Pictures'
    filenames = ['dogs.jpg']
    labels = [1]
    my_dataset = simpleDataset(root, filenames, labels)

    batch_size = 1
    num_workers = 0

    data_loader = data.DataLoader(my_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    for images, labels in data_loader:
        img = transforms.ToPILImage()(images[0])
        plt.imshow(img)
        print(labels)
        plt.show()