import torch
from ENEL645_group6_a2_garbage_model_train import MultimodalGarbageClassifier, GarbageDataset, transform
from torch.utils.data import DataLoader
import os

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
    model = MultimodalGarbageClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    
    # Create test dataset and dataloader
    test_dataset = GarbageDataset(test_dir, transform=transform["test"])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            
            outputs = model(images, text_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on test set: {100 * correct / total:.2f}%') 