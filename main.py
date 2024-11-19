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

# implement operations for our model

def activation(x):
    """
    Implement activation function with tanh()
    :param x: input tensor
    :return: output tensor equals element-wise tanh(x)
    """
    ###############################################################################
    # TODO:                                                                       #
    # 1. calculate act = tanh(x)                                                  #
    ###############################################################################
    # *****BEGIN YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Calculate act = tanh(x)
    act = torch.tanh(x)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return act


def activation_grad(x):
    """
    Calculate the gradient of activation() respect to input x
    You need to find the maths representation of the derivative first
    :param x: input tensor
    :return: element-wise gradient of activation()
    """
    ###############################################################################
    # TODO:                                                                       #
    # 1. find maths represenation of activation()                                 #
    # 2. calculate gradient respect to x                                          #
    ###############################################################################
    # *****BEGIN YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    delta_act = 1 - torch.tanh(x)**2
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return delta_act


def cross_entropy(pred, label):
    """
    Calculate the cross entropy loss, L(pred, label)
    This is for one image only
    :param pred: predicted tensor
    :param label: one-hot encoded label tensor
    :return: the cross entropy loss, L(pred, label)
    """
    ###############################################################################
    # TODO:                                                                       #
    # 1. convert pred into a probability distribution use softmax()               #
    # 2. calculate cross entropy loss                                             #
    ###############################################################################
    # *****BEGIN YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # convert prediction to probabilities
    probs = nn.functional.softmax(pred, dim=0)

    # Cross entropy loss
    loss = -torch.sum(label * torch.log(probs))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


def cross_entropy_grad(pred, label):
    """
    Calculate the gradient of cross entropy respect to pred
    This is for one image only
    :param pred: predicted tensor
    :param label: one-hot encoded label tensor
    :return: gradient of cross entropy respect to pred
    """

    ###############################################################################
    # TODO:                                                                       #
    # 1. calculate element-wise gradient respect to pred = softmax(pred) - label  #
    ###############################################################################
    # *****BEGIN YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Get the softmax
    softmax_p = nn.functional.softmax(pred, dim=0)

    # Get the gradient
    delta_loss = softmax_p - label

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return delta_loss


def forward(w1, b1, w2, b2, x):
    """
    forward operation
    1. one linear layer followed by activation
    2. one linear layer followed by activation
    :param w1:
    :param b1:
    :param w2:
    :param b2:
    :param x: input tensor
    :return: x0, s1, x1, s2, x2
    """
    x0 = x
    ###############################################################################
    # TODO:                                                                       #
    # 1. calculate s1 using w1, x0, b1                                            #
    # 2. calculate x1 using activation()                                          #
    # 3. calculate s2 using w2, x1, b2                                            #
    # 4. calculate x2 using activation()                                          #
    ###############################################################################
    # *****BEGIN YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pre activation value for the first layer
    s1 = torch.matmul(x0, w1.T) + b1

    # activation function for the first layer
    x1 = activation(s1)

    # pre activation value for the second layer
    s2 = torch.matmul(x1, w2.T) + b2

    # activation function for the second layer
    x2 = activation(s2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return x0, s1, x1, s2, x2


def backward(w1, b1, w2, b2, t, x, s1, x1, s2, x2,
             grad_dw1, grad_db1, grad_dw2, grad_db2):
    """
    backward propagation, calculate dl_dw1, dl_db1, dl_dw2, dl_db2 using chain rule
    :param w1:
    :param b1:
    :param w2:
    :param b2:
    :param t: label
    :param x: input tensor
    :param s1:
    :param x1:
    :param s2:
    :param x2:
    :param grad_dw1: gradient of w1
    :param grad_db1: gradient of b1
    :param grad_dw2: gradient of w2
    :param grad_db2: gradient of b2
    :return:
    """
    x0 = x
    ###############################################################################
    # TODO:                                                                       #
    # 1. calculate grad_dx2 using x2, t                                             #
    # 2. calculate grad_ds2 using s2, grad_dx2                                        #
    # 3. calculate grad_dx1 using w2, grad_ds2                                        #
    # 4. calculate grad_ds1 using s1, grad_dx1                                        #
    # 5. calculate and accumulate grad_dw2 using grad_ds2, x1                         #
    # 6. calculate and accumulate grad_db2 using grad_ds2                             #
    # 7. calculate and accumulate grad_dw1 using grad_ds1, x0                         #
    # 8. calculate and accumulate grad_db1 using grad_ds1                             #
    ###############################################################################
    # *****BEGIN YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    grad_dx2 = cross_entropy_grad(x2, t)
    grad_ds2 = grad_dx2 * activation_grad(s2)
    grad_dx1 = torch.matmul(grad_ds2, w2)
    grad_ds1 = grad_dx1 * activation_grad(s1)

    grad_dw2.add_(torch.outer(x1, grad_ds2).T)
    grad_db2.add_(grad_ds2)
    grad_dw1.add_(torch.outer(x0, grad_ds1).T)
    grad_db1.add_(grad_ds1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


