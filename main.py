import torch
from torch import nn
import torchvision
from torchvision import transforms

# load CIFAR-10 dataset with pytorch
# convert to tensor, normalize and flatten
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.Lambda(lambda x: torch.flatten(x)),
])
