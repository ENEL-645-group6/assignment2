import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import os
from transformers import DistilBertModel, DistilBertTokenizer
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
import re
import numpy as np

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

class FiLMClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Image feature extractor (ResNet50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.image_features = nn.Sequential(*list(resnet.children())[:-2])  # Keep spatial dimensions
        
        # Text feature extractor (DistilBERT)
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_drop = nn.Dropout(0.3)
        
        # Freeze DistilBERT parameters
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        # FiLM parameter generators
        self.gamma_generator = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2048)  # Match ResNet feature channels
        )
        
        self.beta_generator = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2048)  # Match ResNet feature channels
        )
        
        # Post-FiLM processing
        self.post_film = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images, input_ids, attention_mask):
        # Get image features (keeping spatial dimensions)
        img_features = self.image_features(images)  # Shape: [B, 2048, H, W]
        
        # Get text features
        text_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]
        text_features = self.text_drop(text_output[:,0])  # Use [CLS] token
        
        # Generate FiLM parameters
        gamma = self.gamma_generator(text_features)  # Shape: [B, 2048]
        beta = self.beta_generator(text_features)    # Shape: [B, 2048]
        
        # Reshape gamma and beta for broadcasting
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)  # [B, 2048, 1, 1]
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)      # [B, 2048, 1, 1]
        
        # Apply FiLM modulation
        modulated_features = (gamma * img_features) + beta
        
        # Process modulated features
        features = self.post_film(modulated_features)
        
        # Classification
        output = self.classifier(features)
        return output


# Custom Dataset class
class ImprovedGarbageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None, max_len=24):
        self.dataset = datasets.ImageFolder(image_folder, transform=transform)
        self.transform = transform
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_len = max_len
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        path = self.dataset.imgs[idx][0]
        filename = os.path.basename(path)
        text = filename.split('.')[0]
        text = ' '.join(filter(lambda x: not x.isdigit(), text.split('_')))
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device=None):
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

            for batch in dataloaders[phase]:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images, input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "film_resnet_adamW_improved_model_best.pth")

    print(f"Best val Acc: {best_acc:.4f}")
    return model


if __name__ == '__main__':
    # Set device
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    
    # Create datasets
    max_len = 24
    data_splits = {
        "train": ImprovedGarbageDataset(train_dir, transform=transform["train"], max_len=max_len),
        "val": ImprovedGarbageDataset(val_dir, transform=transform["val"], max_len=max_len),
        "test": ImprovedGarbageDataset(test_dir, transform=transform["test"], max_len=max_len),
    }

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(data_splits["train"], batch_size=16, shuffle=True, num_workers=2),
        "val": DataLoader(data_splits["val"], batch_size=16, shuffle=False, num_workers=2),
        "test": DataLoader(data_splits["test"], batch_size=16, shuffle=False, num_workers=2),
    }
    
    # Create model
    model = FiLMClassifier(num_classes=4).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), 
                     lr=2e-5,
                     weight_decay=0.01)
    
    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=8, device=device)