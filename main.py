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

# load CIFAR-10 dataset with pytorch
# convert to tensor, normalize and flatten
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.Lambda(lambda x: torch.flatten(x)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_id = list(range(4000))
val_id = list(range(4000, 5000))
test_id = list(range(500))

# subset dataset and create dataloader with batch_size=1
train_sub_set = torch.utils.data.Subset(trainset, train_id)
val_sub_set = torch.utils.data.Subset(trainset, val_id)
test_sub_set = torch.utils.data.Subset(testset, test_id)

train_loader = torch.utils.data.DataLoader(train_sub_set, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_sub_set, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_sub_set, batch_size=1, shuffle=True)

# check data size, should be CxHxW, class map only useful for visualization and sanity checks
image_size = trainset[0][0].size(0)
class_map = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
             9: 'truck'}

