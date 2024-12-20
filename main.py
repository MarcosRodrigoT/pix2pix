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
    "Transform this person into an anime character",
    "Transform this person into an anime character",
    "Transform this person into an anime character",
    "Transform this person into an anime character",
    "Transform this person into a funny sketch",
    "Transform this person into a caricature",
    "Transform this person into a clown",
    "Transform this person into a cowboy",
    (
        "Transform the person into a sinister, evil debtor, radiating an aura of greed and despair. Their face is pale and gaunt, with sunken cheeks, glowing red eyes, and a"
        " twisted, menacing grin revealing sharp, jagged teeth. Their outfit is a tattered, dark suit covered in chains and glowing shackles, representing their eternal debt."
        " Surrounding them is a swirling vortex of crumpled bills, overdue notices, and spectral coins that seem to whisper ominous threats. The background is a shadowy, decrepit"
        " office filled with towering piles of decayed documents and broken calculators, illuminated only by flickering, greenish light emanating from a cursed ledger on their desk."
        " The scene oozes tension and malevolence, capturing the torment of eternal financial doom."
    ),
]

# image = download_image(url)
image = load_image(img_file)

generated_images = pipe(
    prompt,
    image=image,
    num_inference_steps=30,
    image_guidance_scale=1.25,
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
