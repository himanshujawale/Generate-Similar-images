import os
import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- BLIP captioning model ---
blip_path = os.path.join(BASE_DIR, "../Models/models--Salesforce--blip-image-captioning-large"
                                   "/snapshots/353689b859fcf0523410b1806dace5fb46ecdf41")
processor = BlipProcessor.from_pretrained(blip_path)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_path).to(device)
print("BLIP model loaded successfully!")

# --- Stable Diffusion 3 (loaded once) ---
# Adjust this to wherever your SD3 model folder really is
sd_model_path = "stabilityai/stable-diffusion-3.5-medium"

sd_pipe = StableDiffusion3Pipeline.from_pretrained(
    sd_model_path,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    text_encoder_3=None,
    tokenizer_3=None,
)
if device == "cuda":
    sd_pipe.enable_model_cpu_offload()
else:
    sd_pipe.to("cpu")
print("Stable Diffusion model loaded successfully!")


def get_caption(pil_image):
    inputs = processor(images=pil_image.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)


def image_generation(prompt):
    return sd_pipe(
        prompt=prompt,
        negative_prompt="blurred, ugly, watermark, low resolution, blurry",
        num_inference_steps=40,
        height=1024,
        width=1024,
        guidance_scale=9.0,
    ).images[0]


def caption_and_generate(pil_image):
    caption = get_caption(pil_image)
    print("Caption:", caption)
    return image_generation(caption)


demo = gr.Interface(
    fn=caption_and_generate,
    inputs=[gr.Image(label="Select your Image", type="pil")],
    outputs=[gr.Image(label="New Image generated using SD3", type="pil")],
    title="GENAI PROJECT 10: Image Creation Using Stable Diffusion 3.5",
    description="G TO  GENERATE SIMILAR IMAGE BASED ON THE IMAGE UPLOADED.",
)

demo.launch()
