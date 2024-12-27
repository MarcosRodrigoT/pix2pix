import os
import PIL
import torch
import requests
from datetime import datetime
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.thumbnail((400, 400))
    return image


def load_image(path):
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.thumbnail((400, 400))
    return image


model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
img_file = "user_profiles/mrt.jpg"

prompt = [
    "Transform this baby into an anime character",
    "Transform this baby into an anime character",
    "Transform this baby into an anime character",
    "Transform this baby into an anime character",
    "Transform this baby into an anime character",
    "Transform this baby into an anime princess",
    "Transform this baby into an anime princess",
    "Transform this baby into an anime princess",
    "Transform this baby into an anime princess",
    "Transform this baby into an anime princess",
    "Transform this baby into a witch",
    "Transform this baby into a wizard",
    "Transform this baby into a super hero",
    "Transform this baby into a hipopotamus",
]

# image = download_image(url)
image = load_image(img_file)

generated_images = pipe(
    prompt,
    image=image,
    num_inference_steps=60,
    image_guidance_scale=1.5,
)

images = generated_images.images

# Create a directory with the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_dir = os.path.join("images", timestamp)
os.makedirs(output_dir, exist_ok=True)

# Save the images in the created directory
for i, img in enumerate(images):
    img_path = os.path.join(output_dir, f"image_{i}.png")
    img.save(img_path)

print(f"Images saved to {output_dir}")
