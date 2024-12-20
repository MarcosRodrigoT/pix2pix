import os
import PIL
import json
import torch
import requests
from datetime import datetime
from ollama import chat
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


# TODO: Añadir más información de los usuarios. Crear un JSON/YAML.
# Define person info
person_info = {
    "Name": "Carlos",
    "Year of birth": "1993",
    "Place of birth": "Málaga",
}

# Convert dictionary to a string (formatted for readability)
person_info = json.dumps(person_info, indent=2)

# TODO: Checkear el GTI-GPT que hemos creado con una temperatura demasiado alta (20). Aún así los prompts siguen pareciéndose mucho, checkear qué está pasando.
prompt = chat(
    model="GTI-GPT",
    messages=[
        {
            "role": "user",
            "content": (
                "Write a random prompt for a stable diffusion model to generate an image containing a demeaning representation of a person. The topic can be anything, be very"
                " creative. Generate only the prompt, with no additional text or explanation, as I plan to simply copy and paste the response."
            ),
        },
    ],
)

# TODO: Incorporar este script a la app del café.

prompt = prompt["message"]["content"]
print(prompt)

backstory = chat(
    model="llama3.3",
    messages=[
        {
            "role": "user",
            "content": (
                f"Write an imaginary backstory, or lore, for a person that has become an evil debtor. You can use some of the information of that person: {person_info}. Also make"
                f" use of additional information contained in a prompted I used to generate a funny image of this person: {prompt}. Be very specific with details, but overall do not"
                " be too verbose, write a maximum of 2 paragraphs."
            ),
        },
    ],
)

backstory = backstory["message"]["content"]
print(backstory)


# Stop ollama server
os.system("ollama stop llama3.3")

# Construct pipeline
model_id = "stabilityai/stable-diffusion-3.5-large"
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompts = [prompt] * 3

results = pipe(
    prompts,
    num_inference_steps=60,
    guidance_scale=3.5,  # 3.5 - 5.5 -> The bigger this number the more the image will have to resemble the prompt
    height=1024,
    width=1024,
    max_sequence_length=512,
)

images = results.images

# Create a directory with the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_dir = os.path.join("images", timestamp)
os.makedirs(output_dir, exist_ok=True)

# Save the images in the created directory
for i, img in enumerate(images):
    img_path = os.path.join(output_dir, f"image_{i}.png")
    img.save(img_path)

print(f"Images saved to {output_dir}")
