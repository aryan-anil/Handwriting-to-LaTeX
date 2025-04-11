import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from custom_transform import ColorAugmentations


class LaTeXDataset(Dataset):
    def __init__(self, image_folder, label_json_file, tokenizer, transform=None):
         """
        Initializes the LaTeXDataset.

        Args:
            image_folder (str): Path to the folder containing image files.
            label_json_file (str): Path to the JSON file containing image-label mappings.
            tokenizer (GPT2TokenizerFast): GPT2 tokenizer from huggingface transformers.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = tokenizer
        self.augmenter = ColorAugmentations()
        with open(label_json_file, 'r') as f:
            self.labels = json.load(f)
        
        self.image_names = list(self.labels.keys())
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label = self.labels[image_name]
        label = self.tokenizer.bos_token + label
        
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.augmenter.random_augmentation(image)
        """ plt.imshow(image)
        plt.axis('off')
        plt.title("Random Augmented Image")
        plt.show()  """
        if self.transform:
            image = self.transform(image)
        
        
        tokenized_label = self.tokenizer(label, return_tensors='pt', padding=True, truncation=True)
        
        return image, tokenized_label['input_ids'].squeeze(0), tokenized_label['attention_mask'].squeeze(0)


def collate_fn(batch, tokenizer):
    images, input_ids, attention_masks = zip(*batch)
    images = torch.stack(images)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return images, input_ids, attention_masks

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

   
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    
    image_folder = r""      # Folder with images extracted from inkml files
    label_json_file = r""   # json file mapping image name with latex code. {"image.png" : "latex code"}
    dataset = LaTeXDataset(image_folder, label_json_file, tokenizer, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    

    

    print("Sampling 5 encoded labels from the dataset:")
    
    for i,(batch_images, batch_input_ids, batch_attention_masks) in enumerate(dataloader):
       
        for j in range(5):
            input_ids = batch_input_ids[j]
            attention_mask = batch_attention_masks[j]
            
            decoded_text = tokenizer.decode(input_ids)
            print(f"\nSample {j+1}:")
            print(f"Encoded IDs: {input_ids}")
            print(f"Decoded Text: {decoded_text}")
            print(f"Attention Mask: {attention_mask}")
            print("-" * 50)
            
            plt.imshow(batch_images[j].permute(1, 2, 0))
            plt.axis('off')
            plt.show() 

            
