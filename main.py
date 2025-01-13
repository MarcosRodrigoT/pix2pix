import os
import yaml
import json
import torch
from ollama import chat
from diffusers import StableDiffusion3Pipeline


class Color:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


# Construct Stable Diffusion pipeline
model_id = "stabilityai/stable-diffusion-3.5-large"
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# Create a directory for the backstories
backstories_dir = "backstories"
os.makedirs(backstories_dir, exist_ok=True)

# TODO: Añadir más información de los usuarios.
# Load users info
with open("users.yaml", "r") as file:
    data = yaml.safe_load(file)
users = data["users"]

# Iterate over users
for user in users:
    # Create a directory for each user
    user_dir = os.path.join(backstories_dir, user["Name"])
    os.makedirs(user_dir, exist_ok=True)

    # Convert dictionary to a string (formatted for readability)
    user_info = json.dumps(user, indent=2)

    # TODO: It is a known issue that Llama3 is too repetitive. Check other models in hugging face such as "cat llama"
    prompt = chat(
        model="GTI-GPT",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write a random prompt for a stable diffusion model to generate an image containing a demeaning representation of a {user['Gender']} person. The topic"
                    " can be anything, be very creative. Generate only the prompt, with no additional text or explanation, as I plan to simply copy and paste the response."
                ),
            },
        ],
    )
    prompt = prompt["message"]["content"]
    print(f"\n{Color.RED}{prompt}{Color.RESET}")
    with open(f"{user_dir}/prompt.txt", "w") as f:
        f.write(prompt)

    backstory = chat(
        model="llama3.3",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write an imaginary backstory, or lore, for a person that has become an evil debtor. You can use some of the information of that person: {user_info}. Also"
                    f" make use of additional information contained in a prompted I used to generate a funny image of this person: {prompt}. Be very specific with details, but"
                    " overall do not be too verbose, write a maximum of 2 paragraphs."
                ),
            },
        ],
    )
    backstory = backstory["message"]["content"]
    print(backstory)
    with open(f"{user_dir}/backstory.txt", "w") as f:
        f.write(backstory)

    nickname = chat(
        model="llama3.3",
        messages=[
            {
                "role": "user",
                "content": f"Give a short nickname to this person based on his/her backstory: {backstory}. Simply write the nickname, no additional text or explanation.",
            },
        ],
    )
    nickname = nickname["message"]["content"]
    print(f"\n{Color.CYAN}Nickname: {nickname}{Color.RESET}")
    with open(f"{user_dir}/nickname.txt", "w") as f:
        f.write(nickname)

    # Stop ollama server
    os.system("ollama stop llama3.3")

    results = pipe(
        prompt,
        num_inference_steps=60,
        guidance_scale=3.5,  # 3.5 - 5.5 -> The bigger this number the more the image will have to resemble the prompt
        height=1024,
        width=1024,
        max_sequence_length=512,
    )

    # Save the image in the user's directory
    image = results.images[0]
    img_path = os.path.join(user_dir, f"image.png")
    image.save(img_path)

    print(f"{Color.GREEN}Backstory saved to {user_dir}{Color.RESET}")
