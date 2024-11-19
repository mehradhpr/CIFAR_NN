import torch
from torch import nn
import torchvision
from torchvision import transforms

# load CIFAR-10 dataset with pytorch
# convert to tensor, normalize and flatten
transform = transforms.Compose([

    # Convert from PIL format to Tensor
    # Scales pixel values from the range [0, 255] to [0.0, 1.0]
    transforms.ToTensor(),

    # Means and Standard Deviations for the three channels
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),

    # Flattens a 3D image tenser (Color x Height x Width) into a 1D vector
    transforms.Lambda(lambda x: torch.flatten(x)),
])

# Download CIFAR Training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Download CIFAR Testing set
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

# Model Operations -------------------

def activation(x):
    """
    Implement activation function with tanh()
    :param x: input tensor
    :return: output tensor equals element-wise tanh(x)
    """
    # Calculate act = tanh(x)
    act = torch.tanh(x)
    return act


def activation_grad(x):
    """
    Calculate the gradient of activation() respect to input x
    :param x: input tensor
    :return: element-wise gradient of activation()
    """
    delta_act = 1 - torch.tanh(x)**2
    return delta_act


def cross_entropy(pred, label):
    """
    Calculate the cross entropy loss, L(pred, label)
    This is for one image only
    :param pred: predicted tensor
    :param label: one-hot encoded label tensor
    :return: the cross entropy loss, L(pred, label)
    """

    # convert prediction to probabilities
    probs = nn.functional.softmax(pred, dim=0)

    # Cross entropy loss
    loss = -torch.sum(label * torch.log(probs))

    return loss


def cross_entropy_grad(pred, label):
    """
    Calculate the gradient of cross entropy respect to pred
    This is for one image only
    :param pred: predicted tensor
    :param label: one-hot encoded label tensor
    :return: gradient of cross entropy respect to pred
    """

    # Get the softmax
    softmax_p = nn.functional.softmax(pred, dim=0)

    # Get the gradient
    delta_loss = softmax_p - label

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

    # pre activation value for the first layer
    s1 = torch.matmul(x0, w1.T) + b1

    # activation function for the first layer
    x1 = activation(s1)

    # pre activation value for the second layer
    s2 = torch.matmul(x1, w2.T) + b2

    # activation function for the second layer
    x2 = activation(s2)

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

    grad_dx2 = cross_entropy_grad(x2, t)
    grad_ds2 = grad_dx2 * activation_grad(s2)
    grad_dx1 = torch.matmul(grad_ds2, w2)
    grad_ds1 = grad_dx1 * activation_grad(s1)

    grad_dw2.add_(torch.outer(x1, grad_ds2).T)
    grad_db2.add_(grad_ds2)
    grad_dw1.add_(torch.outer(x0, grad_ds1).T)
    grad_db1.add_(grad_ds1)

# training loop, we have 10 classes
nb_classes = 10
nb_train_samples = len(train_loader)

# set number of hidden neurons for first linear layer
nb_hidden = 50
# set learn rate and weights initialization std
lr = 1e-1 / nb_train_samples
init_std = 1e-6

# initialize weights and biases to small values from normal distribution
w1 = torch.empty(nb_hidden, image_size).normal_(0, init_std)
b1 = torch.empty(nb_hidden).normal_(0, init_std)
w2 = torch.empty(nb_classes, nb_hidden).normal_(0, init_std)
b2 = torch.empty(nb_classes).normal_(0, init_std)

# initialize empty tensor for gradients of weights and biases
grad_dw1 = torch.empty(w1.size())
grad_db1 = torch.empty(b1.size())
grad_dw2 = torch.empty(w2.size())
grad_db2 = torch.empty(b2.size())

# run for 1000 epochs
for k in range(1000):

    # initialize loss and train error counts
    acc_loss = 0
    nb_train_errors = 0

    grad_dw1.zero_()
    grad_db1.zero_()
    grad_dw2.zero_()
    grad_db2.zero_()


    for x, y in train_loader:
        train_target_one_hot = nn.functional.one_hot(y.squeeze(dim=0), num_classes=nb_classes)

        # forward propagation
        x0, s1, x1, s2, x2 = forward(w1, b1, w2, b2, x.squeeze(dim=0))

        # prediction
        pred = torch.argmax(x2)

        # accumulate error
        if pred != y:
          nb_train_errors += 1

        # accumulate train loss
        loss = cross_entropy(x2, train_target_one_hot)
        acc_loss += loss.item()

        # backward propogations
        backward(w1, b1, w2, b2, train_target_one_hot, x0, s1, x1, s2, x2,
                 grad_dw1, grad_db1, grad_dw2, grad_db2)

    w1 = w1 - lr * grad_dw1
    b1 = b1 - lr * grad_db1
    w2 = w2 - lr * grad_dw2
    b2 = b2 - lr * grad_db2

    # Val error, initialize val error count
    nb_val_errors = 0

    for x_val, y_val in val_loader:

        x0_val, s1_val, x1_val, s2_val, x2_val = forward(w1, b1, w2, b2,
                                                         x_val.squeeze(dim=0))
        pred_val = torch.argmax(x2_val)
        if pred_val != y_val:
          nb_val_errors += 1

    # print train and val information at end of each epoch
    print('{:d}: acc_train_loss {:.02f}, acc_train_accuracy {:.02f}%, val_accuracy {:.02f}%'
          .format(k,
                  acc_loss,
                  100 - (100 * nb_train_errors) / len(train_loader),
                  100 - (100 * nb_val_errors) / len(val_loader)))
