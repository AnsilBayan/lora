# Install necessary packages
!pip install -q diffusers transformers peft accelerate datasets safetensors bitsandbytes xformers

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer

# Step 1: Load base Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_safetensors=True
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Step 2: (Mock) Apply LoRA weights — replace this with actual LoRA training for real results

# Step 3: Generate images using prompts
def generate_images(subject_token="person001", style="watercolor"):
    prompts = [
        f"{subject_token} in a spacesuit, {style} style",
        f"{subject_token} riding a horse, {style} style",
        f"{subject_token} playing cricket, {style} style"
    ]
    
    images = []
    for prompt in prompts:
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        images.append(image)
    return images

# Step 4: Save results
output_images = generate_images()
for i, img in enumerate(output_images):
    img.save(f"output_scene_{i+1}.png")

print("Images generated and saved successfully.")


