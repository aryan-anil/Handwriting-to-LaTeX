import torch
import torch.nn as nn
from transformers import ViTModel, GPT2Model, GPT2Config

embedding_dim = 768

class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
    def forward(self, x):
        outputs = self.vit(x)
        # Get the [CLS] token embedding
        cls_token = outputs.last_hidden_state[:, 0]  # batch_size x 768
        # Get patch embeddings
        patch_embeddings = outputs.last_hidden_state[:, 1:]  # batch_size x 196 x 768
        return cls_token, patch_embeddings

class GPT2Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=embedding_dim):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_layer=6,  # Reduced to 6 layers
            n_embd=embedding_dim,  
            add_cross_attention=True,
            sequence_length=256
        )
        self.gpt2 = GPT2Model(config)
        self.embedding_dim = embedding_dim
        
    def forward(self, input_ids, attention_mask=None, image_features=None, patch_embeddings=None):
        # Ensure patch_embeddings has the correct shape for encoder_hidden_states
        if patch_embeddings is not None:
            # Add the CLS token to the beginning of patch embeddings
            encoder_hidden_states = torch.cat([image_features.unsqueeze(1), patch_embeddings], dim=1)
        
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False  # Disable past key values for training
        )
        return outputs.last_hidden_state

class ViTGPT2(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vit = ViTFeatureExtractor()
        self.gpt2_decoder = GPT2Decoder(vocab_size=vocab_size)
        
        self.output_projection = nn.Linear(embedding_dim, vocab_size) # 768 x 50257: vocab_size × embedding_dim
    
    def forward(self, images, input_ids, attention_mask=None):
        # Get ViT features
        cls_token, patch_embeddings = self.vit(images)
        
        # Pass through decoder
        decoder_output = self.gpt2_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_features=cls_token,
            patch_embeddings=patch_embeddings
        )
        
        # Project to vocabulary size
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate(self, images, max_length=50, temperature=1.0):
        device = next(self.parameters()).device
        batch_size = images.size(0)
        
        # Get ViT features
        cls_token, patch_embeddings = self.vit(images)
        
        # Initialize with BOS token
        current_ids = torch.full(
            (batch_size, 1), 
            self.gpt2_decoder.gpt2.config.bos_token_id,
            dtype=torch.long, 
            device=device
        )
        
        for _ in range(max_length):
            with torch.no_grad():
                decoder_output = self.gpt2_decoder(
                    input_ids=current_ids,
                    image_features=cls_token,
                    patch_embeddings=patch_embeddings
                )
                logits = self.output_projection(decoder_output)
                
                next_token_logits = logits[:, -1, :] / temperature
                """ next_token = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1), 
                    num_samples=1
                ) """
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(probabilities, dim=-1, keepdim=True)
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                if (next_token == self.gpt2_decoder.gpt2.config.eos_token_id).all():
                    break
                
        return current_ids


