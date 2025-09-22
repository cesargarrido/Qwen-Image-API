import spaces
import gradio as gr
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# Load pipeline at startup
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", 
    torch_dtype=torch.bfloat16
)
pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)

@spaces.GPU(duration=120)
def edit_images(image1, image2, prompt, seed, true_cfg_scale, negative_prompt, num_steps, guidance_scale):
    if image1 is None or image2 is None:
        gr.Warning("Please upload both images")
        return None
    
    # Convert to PIL if needed
    if not isinstance(image1, Image.Image):
        image1 = Image.fromarray(image1)
    if not isinstance(image2, Image.Image):
        image2 = Image.fromarray(image2)
    
    inputs = {
        "image": [image1, image2],
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
    }
    
    with torch.inference_mode():
        output = pipeline(**inputs)
        return output.images[0]

# Example prompts
example_prompts = [
    "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square.",
    "Two characters standing side by side in a beautiful garden with flowers blooming",
    "The hero on the left and the villain on the right, facing off in an epic battle scene",
    "Two friends sitting together on a park bench, enjoying the sunset",
]

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # Qwen Image Edit Plus
        
        Upload two images and describe how you want them combined or edited together.
        
        [Built with anycoder](https://huggingface.co/spaces/akhaliq/anycoder)
        """
    )
    
    with gr.Row():
        with gr.Column():
            image1_input = gr.Image(
                label="First Image",
                type="pil",
                height=300
            )
            image2_input = gr.Image(
                label="Second Image", 
                type="pil",
                height=300
            )
        
        with gr.Column():
            output_image = gr.Image(
                label="Edited Result",
                type="pil",
                height=620
            )
    
    with gr.Group():
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Describe how you want the images combined or edited...",
            value=example_prompts[0],
            lines=3
        )
        
        gr.Examples(
            examples=[[p] for p in example_prompts],
            inputs=[prompt_input],
            label="Example Prompts"
        )
    
    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            seed_input = gr.Number(
                label="Seed",
                value=0,
                precision=0
            )
            num_steps = gr.Slider(
                label="Number of Inference Steps",
                minimum=20,
                maximum=100,
                value=40,
                step=1
            )
        
        with gr.Row():
            true_cfg_scale = gr.Slider(
                label="True CFG Scale",
                minimum=1.0,
                maximum=10.0,
                value=4.0,
                step=0.5
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale",
                minimum=1.0,
                maximum=5.0,
                value=1.0,
                step=0.1
            )
        
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            value=" ",
            placeholder="What to avoid in the generation..."
        )
    
    generate_btn = gr.Button("Generate Edited Image", variant="primary", size="lg")
    
    generate_btn.click(
        fn=edit_images,
        inputs=[
            image1_input,
            image2_input,
            prompt_input,
            seed_input,
            true_cfg_scale,
            negative_prompt,
            num_steps,
            guidance_scale
        ],
        outputs=output_image,
        show_progress="full"
    )

demo.launch()