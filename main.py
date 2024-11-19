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

# Use a larger portion of the dataset
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Increased data size
num_train_total = 50000
num_train_subset = 40000
num_val_subset = 10000

if num_train_total < num_train_subset + num_val_subset:
    raise ValueError("Not enough data in full_trainset for the desired train/val split.")

train_id = list(range(num_train_subset))
val_id = list(range(num_train_subset, num_train_subset + num_val_subset))
test_id = list(range(1000))

train_sub_set = torch.utils.data.Subset(full_trainset, train_id)
val_sub_set = torch.utils.data.Subset(full_trainset, val_id)
test_sub_set = torch.utils.data.Subset(testset, test_id)

train_loader = torch.utils.data.DataLoader(train_sub_set, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_sub_set, batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_sub_set, batch_size=1, shuffle=False)

# check data size
image_size = full_trainset[0][0].size(0)
class_map = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
             9: 'truck'}

# implement operations for our model

def activation_relu(x):
    """
    Implement ReLU activation function
    :param x: input tensor
    :return: output tensor equals element-wise ReLU(x)
    """
    return torch.relu(x)


def activation_relu_grad(x):
    """
    Calculate the gradient of ReLU activation function respect to input x
    :param x: input tensor
    :return: element-wise gradient of ReLU activation
    """
    grad = (x > 0).float()
    return grad


def cross_entropy(pred_logits, label_one_hot):
    """
    Calculate the cross entropy loss from logits.
    :param pred_logits: predicted logits tensor (raw output before softmax)
    :param label_one_hot: one-hot encoded label tensor
    :return: the cross entropy loss
    """
    # Softmax is applied internally here for numerical stability with log
    log_probs = nn.functional.log_softmax(pred_logits, dim=0)
    loss = -torch.sum(label_one_hot * log_probs)
    return loss


def cross_entropy_grad(pred_logits, label_one_hot):
    """
    Calculate the gradient of cross entropy with softmax respect to pred_logits.
    :param pred_logits: predicted logits tensor
    :param label_one_hot: one-hot encoded label tensor
    :return: gradient of cross entropy respect to pred_logits
    """
    softmax_p = nn.functional.softmax(pred_logits, dim=0)
    delta_loss = softmax_p - label_one_hot
    return delta_loss


def forward(w1, b1, w2, b2, x):
    """
    forward operation
    1. one linear layer followed by ReLU activation
    2. one linear layer (output logits)
    :param w1: weights for layer 1
    :param b1: biases for layer 1
    :param w2: weights for layer 2
    :param b2: biases for layer 2
    :param x: input tensor
    :return: x0 (input), s1 (pre-activation L1), x1 (post-activation L1), s2 (logits output)
    """
    x0 = x

    # pre activation value for the first layer
    s1 = torch.matmul(x0, w1.T) + b1

    # activation function for the first layer (ReLU)
    x1 = activation_relu(s1)

    # pre activation value for the second layer (logits)
    s2 = torch.matmul(x1, w2.T) + b2

    # No final activation here, s2 are the logits
    # x2 from previous code is now s2

    return x0, s1, x1, s2 # Removed x2 as it's now s2


def backward(w1, b1, w2, b2, t_one_hot, x0, s1, x1, s2,
             grad_dw1, grad_db1, grad_dw2, grad_db2):
    """
    backward propagation, calculate dl_dw1, dl_db1, dl_dw2, dl_db2 using chain rule
    :param t_one_hot: one-hot label
    :param x0: input tensor
    :param s1: pre-activation values for layer 1
    :param x1: post-activation values for layer 1
    :param s2: logits (output of layer 2)
    :param grad_dw1, grad_db1, grad_dw2, grad_db2: tensors to accumulate gradients
    """

    # Gradient of loss w.r.t. s2 (logits)
    # This is dL/ds2 where L is cross-entropy after softmax(s2)
    grad_L_s2 = cross_entropy_grad(s2, t_one_hot)

    # Gradient of loss w.r.t. weights and biases of layer 2
    grad_dw2.add_(torch.outer(x1, grad_L_s2).T)
    grad_db2.add_(grad_L_s2)

    # Gradient of loss w.r.t. x1 (activation output of layer 1)
    # dL/dx1 = dL/ds2 * ds2/dx1 = grad_L_s2 * w2
    grad_L_x1 = torch.matmul(grad_L_s2, w2)

    # Gradient of loss w.r.t. s1 (pre-activation of layer 1)
    # dL/ds1 = dL/dx1 * dx1/ds1 = grad_L_x1 * activation_relu_grad(s1)
    grad_L_s1 = grad_L_x1 * activation_relu_grad(s1)

    # Gradient of loss w.r.t. weights and biases of layer 1
    grad_dw1.add_(torch.outer(x0, grad_L_s1).T)
    grad_db1.add_(grad_L_s1)

# training loop
nb_classes = 10
nb_train_samples = len(train_loader)

# Model hyperparameters
nb_hidden = 128  # Increased number of hidden neurons
lr = 0.01      # Adjusted learning rate
epochs = 100 # Reduced for quicker testing, you might need 1000 or more

# He initialization for weights (good for ReLU)
std_w1 = torch.sqrt(torch.tensor(2.0 / image_size))
std_w2 = torch.sqrt(torch.tensor(2.0 / nb_hidden))

w1 = torch.empty(nb_hidden, image_size).normal_(0, std_w1)
b1 = torch.zeros(nb_hidden) # Initialize biases to zero
w2 = torch.empty(nb_classes, nb_hidden).normal_(0, std_w2)
b2 = torch.zeros(nb_classes) # Initialize biases to zero

# Tensors for gradients
grad_dw1 = torch.zeros_like(w1)
grad_db1 = torch.zeros_like(b1)
grad_dw2 = torch.zeros_like(w2)
grad_db2 = torch.zeros_like(b2)

print(f"Training with {nb_hidden} hidden neurons, lr={lr}, on {len(train_sub_set)} training samples.")

for k in range(epochs):
    acc_loss = 0
    nb_train_errors = 0

    grad_dw1.zero_()
    grad_db1.zero_()
    grad_dw2.zero_()
    grad_db2.zero_()

    # Training phase
    for x_batch, y_batch in train_loader: # batch_size is 1
        # Squeeze batch dimension as we process one image at a time
        x_sample = x_batch.squeeze(dim=0)
        y_sample = y_batch.squeeze(dim=0)

        train_target_one_hot = nn.functional.one_hot(y_sample, num_classes=nb_classes).float()

        # Forward propagation
        x0, s1, x1, s2_logits = forward(w1, b1, w2, b2, x_sample)

        # Prediction
        pred = torch.argmax(s2_logits)

        if pred != y_sample:
            nb_train_errors += 1

        loss = cross_entropy(s2_logits, train_target_one_hot)
        acc_loss += loss.item()

        # Backward propagation (gradients are accumulated)
        backward(w1, b1, w2, b2, train_target_one_hot, x0, s1, x1, s2_logits,
                 grad_dw1, grad_db1, grad_dw2, grad_db2)

    # Update weights after processing all samples in the epoch
    
    # Averaging gradients over the number of training samples for the epoch update
    w1 -= lr * (grad_dw1 / nb_train_samples)
    b1 -= lr * (grad_db1 / nb_train_samples)
    w2 -= lr * (grad_dw2 / nb_train_samples)
    b2 -= lr * (grad_db2 / nb_train_samples)


    # Validation phase
    nb_val_errors = 0
    val_loss = 0
    with torch.no_grad(): # Disable gradient calculations for validation
        for x_val_batch, y_val_batch in val_loader:
            x_val_sample = x_val_batch.squeeze(dim=0)
            y_val_sample = y_val_batch.squeeze(dim=0)

            _, _, _, s2_logits_val = forward(w1, b1, w2, b2, x_val_sample)
            pred_val = torch.argmax(s2_logits_val)

            if pred_val != y_val_sample:
                nb_val_errors += 1
            
            val_target_one_hot = nn.functional.one_hot(y_val_sample, num_classes=nb_classes).float()
            val_loss += cross_entropy(s2_logits_val, val_target_one_hot).item()


    train_accuracy = 100.0 - (100.0 * nb_train_errors / len(train_loader))
    val_accuracy = 100.0 - (100.0 * nb_val_errors / len(val_loader))
    avg_train_loss = acc_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f'Epoch {k}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

print("Training finished.")

# Testing phase (optional, similar to validation)
# nb_test_errors = 0
# with torch.no_grad():
#     for x_test_batch, y_test_batch in test_loader:
#         x_test_sample = x_test_batch.squeeze(dim=0)
#         y_test_sample = y_test_batch.squeeze(dim=0)
#         _, _, _, s2_logits_test = forward(w1, b1, w2, b2, x_test_sample)
#         pred_test = torch.argmax(s2_logits_test)
#         if pred_test != y_test_sample:
#             nb_test_errors += 1
# test_accuracy = 100.0 - (100.0 * nb_test_errors / len(test_loader))
# print(f'Test Accuracy: {test_accuracy:.2f}%')