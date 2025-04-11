import torch
import torch.nn as nn
import torch.optim as optim
from model import ViTGPT2
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from get_loader import LaTeXDataset, collate_fn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

load_model = False
save_model = True
accumulation_steps = 4
start_epoch=0
num_epochs = 10
log_interval = 1
writer = SummaryWriter(log_dir='logs/vit_gpt2')

def train(model, dataloader, optimizer, criterion, tokenizer, device):
    global start_epoch
    
    if load_model:
        
        checkpoint = torch.load("resnet_gpt2_full_2.pth")
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        """ model.gpt2_decoder.weight.data.uniform_(0.0, 1.0)
        model.gpt2_decoder.bias.data.fill_(0) """

    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        total_correct_tokens = 0
        total_tokens = 0
        total_correct_sequences = 0
        total_sequences = 0
        
        for i, (batch) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
            images, input_ids, attention_masks = batch
            images = images.to(device)            # [batch_size, 3, 256, 256]
            input_ids = input_ids.to(device)      # [batch_size, seq_len]
            attention_masks = attention_masks.to(device) # [batch_size, seq_len]
            
            logits = model(images, input_ids, attention_mask=attention_masks)  # [batch_size, seq_len, vocab_size]
            
            shift_logits = logits[..., :-1, :].contiguous() # [batch_size, seq_len-1, vocab_size]
            shift_labels = input_ids[..., 1:].contiguous()  # [batch_size, seq_len-1]
            
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # [batch_size * (seq_len-1), vocab_size],   [batch_size * (seq_len-1)]

            
            predictions = shift_logits.argmax(dim=-1)  # [batch_size, seq_len-1]
            
            
            mask = shift_labels != tokenizer.pad_token_id  # ignore pad tokens, 1: for pad
            correct_tokens = (predictions == shift_labels) & mask
            total_correct_tokens += correct_tokens.sum().item()
            total_tokens += mask.sum().item()
            
            # Sequence-level accuracy (all tokens in sequence must be correct)
            sequence_correct = torch.all(correct_tokens, dim=1)
            total_correct_sequences += sequence_correct.sum().item()
            total_sequences += sequence_correct.size(0)
            
            # Backward pass and optimization
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            # Log metrics
            if i % log_interval == 0:  
                token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0
                sequence_accuracy = total_correct_sequences / total_sequences if total_sequences > 0 else 0
                
                writer.add_scalar('Loss/train', loss.item() * accumulation_steps, 
                                epoch * len(dataloader) + i)
                writer.add_scalar('Accuracy/token', token_accuracy, 
                                epoch * len(dataloader) + i)
                writer.add_scalar('Accuracy/sequence', sequence_accuracy, 
                                epoch * len(dataloader) + i)
        
        # Compute epoch metrics
        avg_loss = total_loss / len(dataloader)
        token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0
        sequence_accuracy = total_correct_sequences / total_sequences if total_sequences > 0 else 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Token Accuracy: {token_accuracy:.4f}")
        print(f"Sequence Accuracy: {sequence_accuracy:.4f}")
        
        scheduler.step()
        
         
        if epoch % 5 == 0 and save_model and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'vit_gpt2_{epoch}_.pth')

               

if __name__ == "__main__":
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])


    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    device = "cuda:0" 
    vocab_size = len(tokenizer)
    model = ViTGPT2(vocab_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    
    image_folder = r""        # Path to extracted images
    label_json_file = r""     # Path to json file with images to latex code mapping
    dataset = LaTeXDataset(image_folder, label_json_file, tokenizer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer)) 
    train(model, dataloader, optimizer, criterion, tokenizer, device)
        
    
   

