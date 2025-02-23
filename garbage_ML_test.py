import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os


# Define data directories
data_dir = "/Users/rzhang/Desktop/talc_assignment_2"
train_dir = os.path.join(data_dir, "CVPR_2024_dataset_Train")
val_dir = os.path.join(data_dir, "CVPR_2024_dataset_Val")
test_dir = os.path.join(data_dir, "CVPR_2024_dataset_Test")

# Define transformations
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Load datasets
datasets = {
    "train": datasets.ImageFolder(train_dir, transform=transform["train"]),
    "val": datasets.ImageFolder(val_dir, transform=transform["val"]),
    "test": datasets.ImageFolder(test_dir, transform=transform["test"]),
}

# Define data loaders
dataloaders = {
    "train": DataLoader(datasets["train"], batch_size=32, shuffle=True, num_workers=4),
    "val": DataLoader(datasets["val"], batch_size=32, shuffle=False, num_workers=4),
    "test": DataLoader(datasets["test"], batch_size=32, shuffle=False, num_workers=4),
}

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_model.pth")

    print(f"Best val Acc: {best_acc:.4f}")
    return model

# Test function
def test_model(model, dataloader):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Move all the execution code inside the main block
if __name__ == '__main__':
    # Set device to MPS if available (for Apple Silicon), otherwise use CPU
    device = (
        torch.device("mps") 
        if torch.backends.mps.is_available() 
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    
    # Load the pre-trained MobileNetV2 model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Move model to device first
    model = model.to(device)

    # Freeze all layers except the last classifier
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the classifier for 4-class classification
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 4).to(device)  # Move the new layer to device

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

    # Evaluate the model on the test set
    test_model(model, dataloaders["test"])