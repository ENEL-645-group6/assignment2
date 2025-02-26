import torch
##from improved_model_train import ImprovedMultimodalClassifier, ImprovedGarbageDataset, transform
from resnet_model_train import ResNetMultimodalClassifier, ImprovedGarbageDataset, transform
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define data directory
data_dir = "/Users/rzhang/Desktop/talc_assignment_2"
test_dir = os.path.join(data_dir, "CVPR_2024_dataset_Test")

if __name__ == '__main__':
    # Set device priority: CUDA (GPU) > MPS > CPU
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    
    # Create and load model
    model = ResNetMultimodalClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load("resnet_adamW_improved_model_best.pth", weights_only=True))
    model.eval()
    
    # Create test dataset and dataloader
    test_dataset = ImprovedGarbageDataset(test_dir, transform=transform["test"])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Test the model
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
    for i, acc in enumerate(per_class_accuracy):
        print(f'Accuracy for class {i}: {acc:.2f}%') 