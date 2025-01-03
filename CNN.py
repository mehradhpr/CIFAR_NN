import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim

# --- Class and Function Definitions ---

class SimpleCnn(nn.Module):
    def __init__(self, nb_hidden):
        super().__init__()
        # Conv layers with padding
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2) # 32x32 -> 32x32
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv_layer2 = nn.Conv2d(in_channels=32,  out_channels=64, kernel_size=5, padding=2) # After pool1 (10x10) -> 64x10x10
        self.bn2 = nn.BatchNorm2d(64)
        
        # Calculate fc_input_size:
        # Input 32x32
        # Conv1 (k5,p2) -> 32x32. ReLU. BN. Pool1 (k3,s3): floor((32-3)/3 + 1) = 10. -> 32x10x10
        # Conv2 (k5,p2) -> 64x10x10. ReLU. BN. Pool2 (k2,s2): floor((10-2)/2 + 1) = 5. -> 64x5x5
        self.fc_input_size = 64 * 5 * 5 # 1600
        
        self.fully_connected1 = nn.Linear(self.fc_input_size,  nb_hidden)
        self.bn_fc1 = nn.BatchNorm1d(nb_hidden)
        self.dropout_fc = nn.Dropout(0.5) # Dropout layer
        self.fully_connected2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        # Layer 1
        x = self.conv_layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3,  stride=3)
        
        # Layer 2
        x = self.conv_layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten and FC layers
        x = x.view(-1, self.fc_input_size)
        x = self.fully_connected1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fully_connected2(x)
        return x

# train function
def train_model(model, train_loader, val_loader, device, nb_epochs=100, learning_rate=0.001, weight_decay=1e-4): # Added device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    train_samples_count = len(train_loader.dataset)
    val_samples_count = len(val_loader.dataset)

    print(f"Starting training for {nb_epochs} epochs...")
    for e in range(nb_epochs):
        model.train()
        epoch_train_loss_sum = 0
        train_correct_epoch = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device) # Use passed device
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss_sum += loss.item() * x.size(0)
            train_correct_epoch += (pred.argmax(1) == y).type(torch.float).sum().item()

        model.eval()
        epoch_val_loss_sum = 0
        val_correct_epoch = 0
        with torch.no_grad():
            for (x, y) in val_loader:
                x, y = x.to(device), y.to(device) # Use passed device
                pred = model(x)
                loss = criterion(pred, y)
                epoch_val_loss_sum += loss.item() * x.size(0)
                val_correct_epoch += (pred.argmax(1) == y).type(torch.float).sum().item()

        mean_train_loss = epoch_train_loss_sum / train_samples_count
        mean_val_loss = epoch_val_loss_sum / val_samples_count
        acc_train = train_correct_epoch / train_samples_count
        acc_val = val_correct_epoch / val_samples_count
        
        history["train_loss"].append(mean_train_loss)
        history["train_acc"].append(acc_train)
        history["val_loss"].append(mean_val_loss)
        history["val_acc"].append(acc_val)
        
        print(f"EPOCH: {e + 1}/{nb_epochs}")
        print(f"Train loss: {mean_train_loss:.6f}, Train accuracy: {acc_train:.4f}")
        print(f"Val loss: {mean_val_loss:.6f}, Val accuracy: {acc_val:.4f} (LR: {optimizer.param_groups[0]['lr']:.6f})\n")
        
        scheduler.step()
        
    return history
        
def test_model(model, loader, device): # Added device
    print(f"[INFO] Testing model...")
    model.eval()
    test_correct = 0
    total_samples = 0
    with torch.no_grad():
      for x, y in loader:
        x, y = x.to(device), y.to(device) # Use passed device
        pred = model(x)
        test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total_samples += y.size(0)
    accuracy = test_correct / total_samples if total_samples > 0 else 0
    print(f"Accuracy on the set: {accuracy:.4f}")
    return accuracy

# --- Main Execution Block ---
if __name__ == '__main__':
    # select gpu if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Hyperparameters ---
    batch_size = 128
    learning_rate = 0.001
    nb_epochs = 100
    weight_decay_val = 1e-4

    # --- Data Transforms ---
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # --- Load Datasets ---
    full_trainset_for_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    full_trainset_for_validation = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # --- Data Subsets ---
    num_total_train_samples = 50000
    num_train_subset = 45000
    num_val_subset = num_total_train_samples - num_train_subset

    train_indices = list(range(num_train_subset))
    val_indices = list(range(num_train_subset, num_total_train_samples))

    train_sub_set = torch.utils.data.Subset(full_trainset_for_training, train_indices)
    val_sub_set = torch.utils.data.Subset(full_trainset_for_validation, val_indices)

    # --- DataLoaders ---
    train_loader = torch.utils.data.DataLoader(train_sub_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_sub_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- Initialize and Train Model ---
    model = SimpleCnn(nb_hidden=512).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    training_history = train_model(model, train_loader, val_loader, device, nb_epochs=nb_epochs, learning_rate=learning_rate, weight_decay=weight_decay_val)
    
    print("\n--- Final Test ---")
    test_model(model, test_loader, device)

    # --- Plotting Training History ---
    if training_history:
        plt.style.use("ggplot")
        epochs_range = range(1, len(training_history["train_loss"]) + 1)
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, training_history["train_loss"], label="Train Loss")
        plt.plot(epochs_range, training_history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, [acc * 100 for acc in training_history["train_acc"]], label="Train Accuracy")
        plt.plot(epochs_range, [acc * 100 for acc in training_history["val_acc"]], label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc="lower right")

        plt.suptitle("CNN Training History on CIFAR-10")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()