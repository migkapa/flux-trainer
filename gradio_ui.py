import os
import gradio as gr
import tempfile
import zipfile
import shutil
from bfl_finetune import (
    request_finetuning,
    finetune_progress,
    finetune_inference,
    get_inference
)

def create_zip_from_files(files):
    """Create a ZIP file from uploaded files."""
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "training_data.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in files:
            zipf.write(file.name, os.path.basename(file.name))
    
    return zip_path

def start_training(
    files,
    finetune_comment,
    trigger_word,
    mode,
    iterations,
    learning_rate,
    captioning,
    priority,
    finetune_type,
    lora_rank,
    api_key
):
    """Start the training process."""
    if not api_key:
        api_key = os.getenv("BFL_API_KEY")
        if not api_key:
            return "Please provide an API key"
    
    try:
        zip_path = create_zip_from_files(files)
        response = request_finetuning(
            zip_path=zip_path,
            finetune_comment=finetune_comment,
            trigger_word=trigger_word,
            mode=mode,
            api_key=api_key,
            iterations=iterations,
            learning_rate=learning_rate,
            captioning=captioning,
            priority=priority,
            finetune_type=finetune_type,
            lora_rank=lora_rank
        )
        return f"Training started! Finetune ID: {response['id']}"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        # Cleanup
        if 'zip_path' in locals():
            os.remove(zip_path)
            shutil.rmtree(os.path.dirname(zip_path))

def check_progress(finetune_id, api_key):
    """Check training progress."""
    if not api_key:
        api_key = os.getenv("BFL_API_KEY")
        if not api_key:
            return "Please provide an API key"
    
    try:
        result = finetune_progress(finetune_id, api_key)
        return f"Status: {result['status']}\nProgress: {result.get('progress', 'N/A')}"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_image(
    finetune_id,
    prompt,
    finetune_strength,
    endpoint,
    api_key
):
    """Generate an image using the finetuned model."""
    if not api_key:
        api_key = os.getenv("BFL_API_KEY")
        if not api_key:
            return "Please provide an API key"
    
    try:
        # Request inference
        inference_response = finetune_inference(
            finetune_id=finetune_id,
            prompt=prompt,
            finetune_strength=finetune_strength,
            endpoint=endpoint,
            api_key=api_key
        )
        
        # Get the result
        result = get_inference(inference_response['id'], api_key)
        if result['status'] == 'Ready':
            return result['result']['sample']
        else:
            return f"Status: {result['status']}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="FLUX Finetuning UI") as demo:
    gr.Markdown("# FLUX Finetuning Interface")
    
    with gr.Tab("Training"):
        with gr.Row():
            with gr.Column():
                files = gr.File(file_count="multiple", label="Training Images")
                finetune_comment = gr.Textbox(label="Finetune Comment", placeholder="Description of your training")
                trigger_word = gr.Textbox(label="Trigger Word", value="TOK")
                mode = gr.Dropdown(
                    choices=["character", "product", "style", "general"],
                    value="general",
                    label="Mode"
                )
                iterations = gr.Slider(minimum=100, maximum=1000, value=300, step=100, label="Iterations")
                learning_rate = gr.Number(value=0.00001, label="Learning Rate")
                captioning = gr.Checkbox(value=True, label="Enable Captioning")
                priority = gr.Radio(choices=["speed", "quality"], value="quality", label="Priority")
                finetune_type = gr.Radio(choices=["full", "lora"], value="full", label="Finetune Type")
                lora_rank = gr.Radio(choices=[16, 32], value=32, label="LoRA Rank")
                api_key = gr.Textbox(label="API Key (optional if set as environment variable)", type="password")
                train_btn = gr.Button("Start Training")
                train_output = gr.Textbox(label="Training Status")

    with gr.Tab("Progress Check"):
        with gr.Row():
            finetune_id_progress = gr.Textbox(label="Finetune ID")
            api_key_progress = gr.Textbox(label="API Key", type="password")
            check_btn = gr.Button("Check Progress")
            progress_output = gr.Textbox(label="Progress Status")

    with gr.Tab("Generate Images"):
        with gr.Row():
            with gr.Column():
                finetune_id_gen = gr.Textbox(label="Finetune ID")
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                finetune_strength = gr.Slider(minimum=0, maximum=2, value=1.2, label="Finetune Strength")
                endpoint = gr.Dropdown(
                    choices=[
                        "flux-pro-1.1-ultra-finetuned",
                        "flux-pro-finetuned",
                        "flux-pro-1.0-depth-finetuned",
                        "flux-pro-1.0-canny-finetuned",
                        "flux-pro-1.0-fill-finetuned"
                    ],
                    value="flux-pro-1.1-ultra-finetuned",
                    label="Endpoint"
                )
                api_key_gen = gr.Textbox(label="API Key", type="password")
                generate_btn = gr.Button("Generate Image")
            with gr.Column():
                image_output = gr.Image(label="Generated Image")

    # Connect the components
    train_btn.click(
        start_training,
        inputs=[
            files, finetune_comment, trigger_word, mode, iterations,
            learning_rate, captioning, priority, finetune_type, lora_rank, api_key
        ],
        outputs=train_output
    )
    
    check_btn.click(
        check_progress,
        inputs=[finetune_id_progress, api_key_progress],
        outputs=progress_output
    )
    
    generate_btn.click(
        generate_image,
        inputs=[finetune_id_gen, prompt, finetune_strength, endpoint, api_key_gen],
        outputs=image_output
    )

if __name__ == "__main__":
    demo.launch() 