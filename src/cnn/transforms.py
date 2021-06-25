import numpy as np
import torch


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class PIL2numpy:
    def __call__(self, img):
        return np.array(img)


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class OneHot:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def __call__(self, target):
        one_hot_target = np.zeros(self.num_classes)
        one_hot_target[target] = 1
        return one_hot_target
