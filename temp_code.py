import torchvision 

import torch 


if __name__ == '__main__': 
    dataset_train = torchvision.datasets.CIFAR10(root='./data', download=True)
    # print(getStat(dataset_train))