# -*- coding: utf-8 -*-
# @Time    : 2019/9/11 0:10
# @Author  : chaucerhou

import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import os
np.random.seed(2)

def read_csv(path, encoding="utf-8", dtype=str, skiprows=1, delimiter=","):
    if not os.path.exists(path):
        print ("no exists file {}".format(path))
        return None
    data = np.loadtxt(path, dtype=dtype, delimiter=delimiter, skiprows=skiprows)
    return data


class DigitalDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height = 28, width = 28, transforms = None):
        if not os.path.exists(csv_path):
            print("no exists file {}".format(csv_path))
            return None
        print("load file {}".format(csv_path))
        self.data = pd.read_csv(csv_path)
        self.labels = self.data["label"]
        self.data = self.data.drop(labels = ["label"], axis = 1)
        print(self.labels.shape)
        print(self.data.shape)
        self.data = self.data.values.reshape(-1, height, width, 1)
        random_seed = 2
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.data, self.labels, test_size = 0.1, random_state = random_seed)
        self.height = height
        self.width = width
        self.transforms = transforms

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_np = np.asarray(self.data[index]).astype('uint8')
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)

    def change_train(self):
        self.data = self.x_train
        self.labels = self.y_train

    def change_test(self):
        self.data = self.x_val
        self.labels = self.y_val


if __name__ == "__main__":
    transformations = transforms.Compose([transforms.RandomRotation(10),transforms.ToTensor()])
    mnist_from_csv = \
        DigitalDatasetFromCSV('../../../dataset/digit/train.csv', 28, 28, transformations)
    print(len(mnist_from_csv))
    mnist_from_csv.change_train()
    print(len(mnist_from_csv))

