import torch
from diffusers import DiffusionPipeline
import torch.nn as nn
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from SDXL_pipeline import get_image
from SDXL_img2img import image2image

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

# prompt = "The painting captures a serene moment in nature. At the center, a calm lake reflects the sky, its surface rippled only by the gentlest of breezes. The sky above is a brilliant mix of blues and whites, with fluffy clouds drifting leisurely across. On the banks of the lake, tall trees stand gracefully, their leaves rustling in the wind. In the foreground, an old man sits on a rock, seemingly lost in deep thought or meditation. The soft light of the setting sun bathes the entire scene in a warm glow, creating a sense of peace and tranquility. The colors are muted yet vibrant, and the details are captured with precision, giving the painting a sense of realism while still retaining a dreamlike quality."
# prompt = "The image captures a breathtaking view of the Hong Kong skyline at sunset. The sky, awash with hues of orange and blue, serves as a stunning backdrop to the city's architectural marvels. The tallest building, the International Finance Centre, pierces the sky with its imposing height. Its lights are switched on, casting a warm glow that contrasts with the cool tones of the evening sky.  The other buildings, though not as tall, are no less impressive. They are adorned with lights that twinkle like stars against the twilight sky. The water below mirrors the sky's colors, adding to the overall vibrancy of the scene.  The perspective of the image is from the water, looking towards the shore. This viewpoint allows for a comprehensive view of the cityscape, from the towering skyscrapers to the smaller structures nestled among them. The image encapsulates the essence of Hong Kong's urban landscape, a blend of modernity and natural beauty."
# prompt = "This picture shows a quiet and beautiful park. In distance, there is a mountain that reaches to the sky. In the park, a deep path leads to a distant forest. The trees on both sides of the road are maples, and yellow maple leaves are falling on the path, award-winning, professional, highly detailed."
# prompt = "The afternoon sun casts diagonal rays onto the alley, creating intertwining shadows on the cobblestone path. An antique street lamp stands in a corner, patiently awaiting the arrival of dusk. An old-fashioned bicycle is parked against the wall, seemingly waiting for its next rider. The Alley is filled with tranquility, a testament to the stillness of time."
# prompt = "A brown dog sits under a big oak tree, looking up at the bright sun. Nearby, a small gray cat sleeps on a patch of soft grass beside colorful flowers. The flowers are bright red and yellow, swaying gently in the warm breeze. The dog watches the cat as it rests peacefully under the tree’s cool shade. Above, the sun shines high in the blue sky, casting long shadows from the tree."
# prompt = "A tall man stands next to a woman in a sunny park, holding a bunch of red flowers. The woman, with long brown hair, smiles as she reaches for the flowers. A young kid, wearing a blue shirt, runs around them, laughing and playing. The sun shines brightly above, casting warm light on the scene. The flowers in the man’s hand sway gently in the soft breeze. The woman bends down to show the kid the flowers, and they both smile."
prompt = "A cat on the left of a dog."

image = get_image(
    pipe=base,
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
    
    
image = image2image(
    pipe=refiner,
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]
    
image_name = "sdxl.png"
image.save(image_name)
