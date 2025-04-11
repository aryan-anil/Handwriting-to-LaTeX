import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from model import ViTGPT2
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


image_path = r""  # path to image
model_path = r""  # path to model

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
vocab_size  = len(tokenizer) # 50257
model = ViTGPT2(50257).to("cuda:0")

checkpoint = torch.load(model_path)
#print(checkpoint.keys())
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).to("cuda:0").unsqueeze(0)

img_tensor1 = img_tensor.squeeze(0).cpu()  
print(img_tensor1.shape)           #torch.Size([3, 256, 256])

# Convert the tensor to a PIL image
to_pil = ToPILImage()  
img = to_pil(img_tensor1)

#img.save("image1.jpg")

plt.imshow(img)
plt.axis('off')  
plt.show()

generated_ids = model.generate(
                    images=img_tensor,
                    temperature=1
                ) 
                
                
predicted_text = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=False)
print(f"Predicted code: {predicted_text}") 
