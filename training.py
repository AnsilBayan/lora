# Install required libraries
#pip install -q diffusers transformers accelerate peft datasets safetensors bitsandbytes xformers

# Step 1: Import libraries
import os
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
from transformers import CLIPTokenizer
from PIL import Image
import torch

# Step 2: Set paths and hyperparameters
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
instance_prompt = "a photo of sks person"
output_dir = "C://Users//ACER//Desktop//lora"
train_data_dir = "C://Users//ACER//Desktop//face_images"  # Directory with your 3–4 face images

# Step 3: Prepare dataset
from torchvision import transforms
from torch.utils.data import Dataset
import glob

class FaceDataset(Dataset):
    def _init_(self, image_dir, tokenizer, instance_prompt):
        self.image_paths = glob.glob(f"{image_dir}/*.jpg")
        self.tokenizer = tokenizer
        self.prompt = instance_prompt
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        example = {}
        example["pixel_values"] = self.transform(image)
        example["input_ids"] = self.tokenizer(self.prompt, return_tensors="pt").input_ids[0]
        return example

# Step 4: Load tokenizer and prepare dataset
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path)
train_dataset = FaceDataset(train_data_dir, tokenizer, instance_prompt)

# Step 5: Define LoRA Config
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["attn1", "attn2"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.TEXT_TO_IMAGE
)

# Step 6: Prepare training
from torch.utils.data import DataLoader
from tqdm import tqdm

pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16).to("cuda")
unet = pipeline.unet
unet = get_peft_model(unet, lora_config)

accelerator = Accelerator()
unet, train_dataloader = accelerator.prepare(unet, DataLoader(train_dataset, batch_size=1, shuffle=True))

optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

# Step 7: Training loop
unet.train()
for epoch in range(5):
    for batch in tqdm(train_dataloader):
        pixel_values = batch["pixel_values"].unsqueeze(0).to("cuda")
        optimizer.zero_grad()
        noise = torch.randn_like(pixel_values)
        loss = ((unet(pixel_values + noise).sample - pixel_values)**2).mean()
        accelerator.backward(loss)
        optimizer.step()

# Step 8: Save LoRA weights
unet.save_pretrained(output_dir)
print("LoRA training completed and model saved.")


