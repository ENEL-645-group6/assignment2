
import torch
import torch.utils.data
from torchvision import datasets
from transformers import DistilBertTokenizer
import os

class ImageTextGarbageDataset(torch.utils.data.Dataset):
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