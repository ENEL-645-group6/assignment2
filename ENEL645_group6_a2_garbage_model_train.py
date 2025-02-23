import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import os
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.utils.data import DataLoader

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

class MultimodalGarbageClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Image feature extractor (MobileNetV2)
        self.image_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.image_features = self.image_model.features
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Text feature extractor (BERT)
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT parameters (optional)
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        # Fusion and classification layers
        self.image_fc = nn.Linear(1280, 512)  # MobileNetV2 features
        self.text_fc = nn.Linear(768, 512)    # BERT features
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # 512 + 512 = 1024
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, text_inputs):
        # Process images
        img_features = self.image_features(images)
        img_features = self.image_pool(img_features)
        img_features = img_features.view(img_features.size(0), -1)
        img_features = self.image_fc(img_features)

        # Process text
        text_outputs = self.text_model(**text_inputs)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_features = self.text_fc(text_features)

        # Combine features
        combined_features = torch.cat((img_features, text_features), dim=1)
        output = self.classifier(combined_features)
        return output

# Custom Dataset class
class GarbageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.dataset = datasets.ImageFolder(image_folder, transform=transform)
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Get the image filename
        path = self.dataset.imgs[idx][0]
        filename = os.path.basename(path)
        # Process filename to get descriptive text (remove extension and numbers)
        text = filename.split('.')[0]  # Remove extension
        text = ' '.join(filter(lambda x: not x.isdigit(), text.split('_')))  # Remove numbers
        
        # Tokenize text
        text_encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=32,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'text_inputs': {
                'input_ids': text_encoding['input_ids'].squeeze(0),
                'attention_mask': text_encoding['attention_mask'].squeeze(0)
            },
            'label': label
        }

# Modified DataLoader creation
data_splits = {
    "train": GarbageDataset(train_dir, transform=transform["train"]),
    "val": GarbageDataset(val_dir, transform=transform["val"]),
    "test": GarbageDataset(test_dir, transform=transform["test"]),
}

dataloaders = {
    "train": DataLoader(data_splits["train"], batch_size=32, shuffle=True, num_workers=2),
    "val": DataLoader(data_splits["val"], batch_size=32, shuffle=False, num_workers=2),
    "test": DataLoader(data_splits["test"], batch_size=32, shuffle=False, num_workers=2),
}

# Modified training function
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

            for batch in dataloaders[phase]:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images, text_inputs)
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
                torch.save(model.state_dict(), "best_model.pth")

    print(f"Best val Acc: {best_acc:.4f}")
    return model

# Main execution
if __name__ == '__main__':
    # Set device priority: CUDA (GPU) > MPS > CPU
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    
    # Create model
    model = MultimodalGarbageClassifier(num_classes=4).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5)