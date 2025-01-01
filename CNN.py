import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# select gpu if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load CIFAR-10 dataset with pytorch
# set batch_size
batch_size = 100
# convert to tensor, normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_id = list(range(4000))
val_id = list(range(4000, 5000))
test_id = list(range(500))

# subset dataset and create dataloader with batch_size
train_sub_set = torch.utils.data.Subset(trainset, train_id)
val_sub_set = torch.utils.data.Subset(trainset, val_id)
test_sub_set = torch.utils.data.Subset(testset, test_id)

train_loader = torch.utils.data.DataLoader(train_sub_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_sub_set, batch_size=batch_size, shuffle=True) # shuffle=True for val_loader is unusual, usually False
test_loader = torch.utils.data.DataLoader(test_sub_set, batch_size=batch_size, shuffle=True) # shuffle=True for test_loader is unusual, usually False

# check data size, should be (C,H,W), class map only useful for visualization and sanity checks
image_size = trainset[0][0].size()
class_map = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
             9: 'truck'}

class SimpleCnn(nn.Module):
    def __init__(self, nb_hidden):
        super().__init__()

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5 )
        self.conv_layer2 = nn.Conv2d(in_channels=32,  out_channels=64, kernel_size=5)
        # Calculate the input size to the first fully connected layer
        # After conv1: (32-5+1)/1 = 28. After pool1 (k3,s3): floor((28-3)/3 + 1) = 9. So, 32x9x9
        # After conv2: (9-5+1)/1 = 5. After pool2 (k2,s2): floor((5-2)/2 + 1) = 2. So, 64x2x2 = 256
        self.fc_input_size = 64 * 2 * 2 # This is 256
        self.fully_connected1 = nn.Linear(self.fc_input_size,  nb_hidden)
        self.fully_connected2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        """
        forward step
        :param x: input tensor
        :return: output tensor
        """
        c1_out = F.max_pool2d(F.relu(self.conv_layer1(x)), kernel_size=3,  stride=3)
        c2_out = F.max_pool2d(F.relu(self.conv_layer2(c1_out)), kernel_size=2, stride=2)
        c2_out = c2_out.view(-1, self.fc_input_size) # Use calculated fc_input_size
        fc1_out = F.relu( self.fully_connected1(c2_out) )
        fc2_out = self.fully_connected2(fc1_out)
        return fc2_out

# train function
def train_model(model, train_loader, val_loader, nb_epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters()) # Adam optimizer

    # initialize loss/acc dict storage
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    # Calculate steps based on DataLoader length (number of batches)
    # This is more robust if drop_last=False and last batch is smaller
    train_steps_per_epoch = len(train_loader)
    val_steps_per_epoch = len(val_loader)


    # run for nb_epochs
    for e in range(nb_epochs):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss for the epoch
        epoch_train_loss_sum = 0 # Sum of losses for the epoch
        epoch_val_loss_sum = 0   # Sum of losses for the epoch
        # initialize the number of correct predictions in the training
        # and validation step
        train_correct_epoch = 0
        val_correct_epoch = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss (item() gives Python number, good for summing)
            epoch_train_loss_sum += loss.item()
            train_correct_epoch += (pred.argmax(1) == y).type(torch.float).sum().item()

        # switch off autograd for validation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                epoch_val_loss_sum += loss.item()
                val_correct_epoch += (pred.argmax(1) == y).type(torch.float).sum().item()

        # calculate the average epoch training and validation loss
        mean_train_loss = epoch_train_loss_sum / train_steps_per_epoch # Average loss per batch
        mean_val_loss = epoch_val_loss_sum / val_steps_per_epoch     # Average loss per batch
        
        # calculate the training and validation accuracy
        # len(train_loader.dataset) is the total number of samples in the subset
        acc_train = train_correct_epoch / len(train_loader.dataset)
        acc_val = val_correct_epoch / len(val_loader.dataset)
        
        # update our training history
        history["train_loss"].append(mean_train_loss)
        history["train_acc"].append(acc_train) # Store as 0.0-1.0
        history["val_loss"].append(mean_val_loss)
        history["val_acc"].append(acc_val)   # Store as 0.0-1.0
        
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, nb_epochs))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
            mean_train_loss, acc_train)) # acc_train is 0.0-1.0
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            mean_val_loss, acc_val))   # acc_val is 0.0-1.0
        
    return history # Return the history dictionary
        
def test(model, test_loader):
    # we can now evaluate the network on the test set
    print("[INFO] testing SimpleCnn...")
    # turn off autograd for testing evaluation

    test_correct = 0
    total_samples = 0 # Use a more descriptive name

    # Here we switch off autograd
    with torch.no_grad():
      # set the evaluation mode
      model.eval()
      # looping over the test loader
      for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        # get the prediction for batch x
        pred = model(x)
        test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_samples += y.size(0) # Accumulate total samples processed

    average_test_accuracy = test_correct / total_samples if total_samples > 0 else 0

    # print the average test accuracy
    print(f"Test Accuracy: {average_test_accuracy:.4f}") # Display as 0.0-1.0

model = SimpleCnn(nb_hidden=500).to(device)

# Call train_model and store the returned history
training_history = train_model(model, train_loader, val_loader, nb_epochs=300) # nb_epochs was 300 in your call
test(model, test_loader)

# --- Plotting Training History ---
if training_history:
    plt.style.use("ggplot")

    epochs_range = range(1, len(training_history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, training_history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [acc * 100 for acc in training_history["train_acc"]], label="Train Accuracy") # Multiply by 100 for percentage
    plt.plot(epochs_range, [acc * 100 for acc in training_history["val_acc"]], label="Validation Accuracy") # Multiply by 100 for percentage
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")

    plt.suptitle("Training History")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()