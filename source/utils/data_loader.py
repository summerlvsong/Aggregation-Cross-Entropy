import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_name, length, class_num, transform=None):
        """
        Args:
            file_name (string): Path to the files with images and their annotations.
            length (string): image number.
            class_num (int): class number.
        """
        with open(file_name) as fh:
            self.img_and_label = fh.readlines()
        self.length = length
        self.transform = transform
        self.class_num = class_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img_and_label = self.img_and_label[idx].strip()
        pth, word = img_and_label.split(' ') # image path and its annotation

        image = cv2.imread(pth,0)
        image = cv2.pyrDown(image).astype('float32') # 100*100

        word = [ord(var)-97 for var in word] # a->0

        label = np.zeros((self.class_num+1)).astype('float32')

        for ln in word:
            label[int(ln+1)] += 1 # label construction for ACE

        label[0] = len(word)

        sample = {'image': image, 'label': label}

        sample = {'image': torch.from_numpy(image).unsqueeze(0), 'label': torch.from_numpy(label)}

        if self.transform:
            sample = self.transform(sample)

        return sample    


