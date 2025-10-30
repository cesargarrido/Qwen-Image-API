from fastapi import FastAPI, Request
from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image
import torch, math, requests, base64
from io import BytesIO

app = FastAPI()

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    scheduler=scheduler,
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
pipeline.load_lora_weights(
    "lightx2v/Qwen-Image-Lightning",
    weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
)
pipeline.fuse_lora()

@app.post("/infer")
async def infer(request: Request):
    data = await request.json()
    image1_url = data["image1"]
    image2_url = data["image2"]
    prompt = data["prompt"]
    seed = data.get("seed", 0)
    true_cfg_scale = data.get("true_cfg_scale", 1.0)
    negative_prompt = data.get("negative_prompt", "")
    num_steps = data.get("num_steps", 8)
    guidance_scale = data.get("guidance_scale", 1.0)

    # Descarga imágenes
    img1 = Image.open(BytesIO(requests.get(image1_url).content))
    img2 = Image.open(BytesIO(requests.get(image2_url).content))

    # Generación
    with torch.inference_mode():
        output = pipeline(
            image=[img1, img2],
            prompt=prompt,
            generator=torch.manual_seed(seed),
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )

    img = output.images[0]
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {"image_base64": img_b64}
