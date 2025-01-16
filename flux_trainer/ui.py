"""FLUX Trainer UI module."""

import base64
import logging
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Tuple, Union, Dict

import gradio as gr
from PIL import Image

from flux_trainer.api import FluxAPI, FluxAPIError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
REFRESH_INTERVAL = 5  # seconds
AUTO_REFRESH_STATES = {"Pending", "Running", "Processing"}  # States that should trigger auto-refresh


def create_zip_from_files(files: List[str]) -> Tuple[str, int]:
    """Create a base64 encoded ZIP file from the provided files."""
    if not files:
        raise ValueError("No files provided")

    total_size = sum(os.path.getsize(f.name) for f in files)
    logger.info(f"Creating ZIP from {len(files)} files, total size: {total_size/1024/1024:.2f}MB")

    if total_size > 100 * 1024 * 1024:
        raise ValueError("Total file size exceeds 100MB limit")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip.name, "w") as zipf:
            for file in files:
                logger.debug(f"Adding file to ZIP: {file.name}")
                zipf.write(file.name, Path(file.name).name)
        
        with open(temp_zip.name, "rb") as f:
            zip_data = base64.b64encode(f.read()).decode("utf-8")
    
    os.unlink(temp_zip.name)
    return zip_data, total_size


class FluxTrainerUI:
    """Gradio-based UI for FLUX training and inference."""

    def __init__(self):
        """Initialize the UI components."""
        self.api = None
        self.demo = gr.Blocks(title="FLUX Trainer")
        self.default_api_key = os.getenv("BFL_API_KEY", "")
        self.default_description = os.getenv("FLUX_DEFAULT_DESCRIPTION", "")
        self._build_interface()
        logger.info("UI initialized")

    def _build_interface(self):
        """Build the Gradio interface."""
        with self.demo:
            gr.Markdown("# FLUX Trainer")
            gr.Markdown("Train and use custom image generation models with FLUX.")
            
            # Store state for auto-refresh
            auto_refresh = gr.State(False)
            last_status = gr.State("")
            
            # Define helper functions first
            def clear_on_error(error_msg: str) -> str:
                """Clear error message after a delay."""
                if error_msg and error_msg.startswith("Error:"):
                    time.sleep(3)  # Show error for 3 seconds
                    return ""
                return error_msg
            
            def should_continue(auto_refresh: bool, last_status: str) -> bool:
                """Determine if auto-refresh should continue."""
                return auto_refresh and last_status in AUTO_REFRESH_STATES
            
            def auto_refresh_list(api_key: str) -> List[List[str]]:
                """Auto-refresh the finetunes list."""
                try:
                    return list_finetunes(api_key or self.default_api_key)
                except Exception:
                    return []  # Silently fail on auto-refresh errors
            
            # Build interface components
            with gr.Tab("Training"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Upload Training Images")
                        gr.Markdown("Upload 1-20 images (max 100MB total)")
                        files = gr.File(
                            file_count="multiple",
                            label="Training Images",
                            file_types=["image"]
                        )
                        
                        gr.Markdown("### Training Configuration")
                        finetune_comment = gr.Textbox(
                            label="Description",
                            placeholder="Describe your training dataset",
                            value=self.default_description
                        )
                        if self.default_description:
                            gr.Markdown("*✓ Using default description from environment*")
                        gr.Markdown("*A description to help identify this model*")
                        
                        trigger_word = gr.Textbox(
                            label="Trigger Word",
                            value="TOK"
                        )
                        gr.Markdown("*Word to use in prompts to trigger your trained concept*")
                        
                        mode = gr.Dropdown(
                            choices=["character", "product", "style", "general"],
                            value="general",
                            label="Training Mode"
                        )
                        gr.Markdown("*How to interpret your training images*")
                        
                        iterations = gr.Slider(
                            minimum=100,
                            maximum=1000,
                            value=300,
                            step=100,
                            label="Training Iterations"
                        )
                        gr.Markdown("*More iterations = better results but longer training*")
                        
                        learning_rate = gr.Number(
                            value=0.00001,
                            label="Learning Rate",
                            precision=5
                        )
                        gr.Markdown("*Lower values are more stable but train slower*")
                        
                        captioning = gr.Checkbox(
                            value=True,
                            label="Enable Auto-captioning"
                        )
                        gr.Markdown("*Automatically generate captions for training images*")
                        
                        priority = gr.Radio(
                            choices=["speed", "quality"],
                            value="quality",
                            label="Training Priority"
                        )
                        gr.Markdown("*Quality is recommended for best results*")
                        
                        finetune_type = gr.Radio(
                            choices=["full", "lora"],
                            value="full",
                            label="Finetuning Type"
                        )
                        gr.Markdown("*Full finetuning is more powerful but slower*")
                        
                        lora_rank = gr.Radio(
                            choices=[16, 32],
                            value=32,
                            label="LoRA Rank"
                        )
                        gr.Markdown("*Higher rank = better quality but slower training*")
                        
                        api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Enter to override environment API key",
                            value=self.default_api_key
                        )
                        if self.default_api_key:
                            gr.Markdown("*✓ Using API key from environment*")
                        else:
                            gr.Markdown("*⚠️ No API key found in environment*")
                        
                        train_btn = gr.Button("Start Training", variant="primary")
                        train_output = gr.Textbox(
                            label="Training Status",
                            interactive=False
                        )

            with gr.Tab("Progress"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Monitor Training Progress")
                        finetune_id_progress = gr.Textbox(
                            label="Finetune ID",
                            placeholder="Enter the ID from the training step"
                        )
                        api_key_progress = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Enter to override environment API key",
                            value=self.default_api_key
                        )
                        with gr.Row():
                            check_btn = gr.Button("Check Progress", variant="primary")
                            auto_refresh_toggle = gr.Checkbox(
                                label="Auto Refresh",
                                value=True
                            )
                            gr.Markdown(f"*Automatically refresh every {REFRESH_INTERVAL} seconds*")
                        progress_output = gr.Textbox(
                            label="Progress Status",
                            interactive=False
                        )

            with gr.Tab("Manage"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Manage Finetunes")
                        api_key_manage = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Enter to override environment API key",
                            value=self.default_api_key
                        )
                        list_btn = gr.Button("List Finetunes", variant="primary")
                        finetunes_output = gr.Dataframe(
                            headers=["ID", "Comment", "Status", "Created"],
                            label="Your Finetunes",
                            interactive=False
                        )
                        delete_id = gr.Textbox(
                            label="Finetune ID to Delete",
                            placeholder="Enter ID from the list above"
                        )
                        delete_btn = gr.Button("Delete Finetune", variant="secondary")
                        manage_output = gr.Textbox(
                            label="Operation Status",
                            interactive=False
                        )

            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Generate Images")
                        finetune_id_gen = gr.Textbox(
                            label="Finetune ID",
                            placeholder="Enter the ID of your trained model"
                        )
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your image generation prompt"
                        )
                        gr.Markdown("*Include your trigger word to use the trained concept*")
                        
                        finetune_strength = gr.Slider(
                            minimum=0,
                            maximum=2,
                            value=1.2,
                            label="Finetune Strength"
                        )
                        gr.Markdown("*Higher values = stronger effect of your training*")
                        
                        endpoint = gr.Dropdown(
                            choices=[
                                "flux-pro-1.1-ultra-finetuned",
                                "flux-pro-finetuned",
                                "flux-pro-1.0-depth-finetuned",
                                "flux-pro-1.0-canny-finetuned",
                                "flux-pro-1.0-fill-finetuned"
                            ],
                            value="flux-pro-1.1-ultra-finetuned",
                            label="Model Endpoint"
                        )
                        gr.Markdown("*The base model to use*")
                        
                        api_key_gen = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Enter to override environment API key",
                            value=self.default_api_key
                        )
                        generate_btn = gr.Button("Generate Image", variant="primary")
                    with gr.Column():
                        image_output = gr.Image(
                            label="Generated Image",
                            type="filepath"
                        )

            def check_progress_with_state(finetune_id: str, api_key: str, auto_refresh: bool, last_status: str) -> Tuple[str, bool, str]:
                """Check progress and manage auto-refresh state."""
                try:
                    if not finetune_id:
                        return "Error: Please enter a Finetune ID", False, ""
                    
                    api = self._get_api(api_key)
                    result = api.get_progress(finetune_id)
                    
                    status = result['status']
                    progress = result.get('progress', 'N/A')
                    
                    # Determine if we should continue auto-refresh
                    should_refresh = status in AUTO_REFRESH_STATES
                    
                    # Format the output message
                    output = (
                        f"Status: {status}\n"
                        f"Progress: {progress}\n"
                    )
                    
                    if status == "Ready":
                        output += "✅ Training complete! You can now use the Generate tab."
                    elif status in AUTO_REFRESH_STATES:
                        output += f"🔄 Auto-refreshing every {REFRESH_INTERVAL} seconds..."
                    elif status == "Failed":
                        output += "❌ Training failed. Please check your parameters and try again."
                    
                    return output, should_refresh, status
                    
                except FluxAPIError as e:
                    logger.error(f"API error checking progress: {str(e)}")
                    return f"API Error: {str(e)}", False, ""
                except Exception as e:
                    logger.error(f"Unexpected error checking progress: {str(e)}")
                    return f"Error: {str(e)}", False, ""

            def list_finetunes(api_key: str) -> List[List[str]]:
                """Handle listing finetunes."""
                try:
                    api = self._get_api(api_key)
                    response = api.list_finetunes()
                    
                    # Extract finetunes list from response
                    finetunes = response.get('finetunes', [])
                    logger.info(f"Found {len(finetunes)} finetunes")
                    
                    # Format the data for display
                    rows = []
                    for ft in finetunes:
                        try:
                            # Get details for each finetune
                            ft_id = ft if isinstance(ft, str) else ft.get('id')
                            if not ft_id:
                                logger.warning(f"Invalid finetune data: {ft}")
                                continue
                                
                            details = api.get_finetune_details(ft_id)
                            if isinstance(details, dict):
                                rows.append([
                                    ft_id,
                                    details.get('finetune_comment', 'N/A'),
                                    details.get('status', 'N/A'),
                                    details.get('created_at', 'N/A')
                                ])
                            else:
                                logger.warning(f"Invalid details format for {ft_id}: {details}")
                                rows.append([
                                    ft_id,
                                    'Error: Invalid details format',
                                    'N/A',
                                    'N/A'
                                ])
                        except Exception as e:
                            logger.error(f"Error getting details for finetune {ft}: {str(e)}")
                            ft_id = ft if isinstance(ft, str) else ft.get('id', 'Unknown')
                            rows.append([
                                ft_id,
                                'Error getting details',
                                'N/A',
                                'N/A'
                            ])
                    
                    if not rows:
                        return [["No finetunes found", "", "", ""]]
                    
                    return rows
                    
                except FluxAPIError as e:
                    logger.error(f"API error listing finetunes: {str(e)}")
                    return [[f"API Error: {str(e)}", "", "", ""]]
                except Exception as e:
                    logger.error(f"Unexpected error listing finetunes: {str(e)}")
                    return [[f"Error: {str(e)}", "", "", ""]]

            def delete_finetune(finetune_id: str, api_key: str) -> str:
                """Handle deleting a finetune."""
                try:
                    if not finetune_id:
                        return "Error: Please enter a Finetune ID"
                    
                    api = self._get_api(api_key)
                    api.delete_finetune(finetune_id)
                    return f"Successfully deleted finetune: {finetune_id}"
                    
                except FluxAPIError as e:
                    logger.error(f"API error deleting finetune: {str(e)}")
                    return f"API Error: {str(e)}"
                except Exception as e:
                    logger.error(f"Unexpected error deleting finetune: {str(e)}")
                    return f"Error: {str(e)}"

            # Helper function to get effective API key
            def get_api_key(ui_key: str) -> str:
                """Get effective API key, preferring UI input over environment."""
                return ui_key if ui_key else self.default_api_key

            # Modify handlers to use the helper
            def handle_training_with_key(*args):
                api_key = args[-1]  # API key is last argument
                return self._handle_training(*args[:-1], get_api_key(api_key))

            def handle_progress_with_key(finetune_id: str, api_key: str, auto_refresh: bool, last_status: str):
                return check_progress_with_state(finetune_id, get_api_key(api_key), auto_refresh, last_status)

            def handle_generation_with_key(*args):
                api_key = args[-1]  # API key is last argument
                return self._handle_generation(*args[:-1], get_api_key(api_key))

            def handle_list_finetunes(api_key: str):
                return list_finetunes(get_api_key(api_key))

            def handle_delete_finetune(finetune_id: str, api_key: str):
                return delete_finetune(finetune_id, get_api_key(api_key))

            # Connect event handlers with modified key handling
            train_btn.click(
                handle_training_with_key,
                inputs=[
                    files, finetune_comment, trigger_word, mode, iterations,
                    learning_rate, captioning, priority, finetune_type,
                    lora_rank, api_key
                ],
                outputs=train_output
            ).then(
                clear_on_error,
                inputs=train_output,
                outputs=train_output,
                show_progress=False
            )
            
            check_progress_event = check_btn.click(
                handle_progress_with_key,
                inputs=[finetune_id_progress, api_key_progress, auto_refresh, last_status],
                outputs=[progress_output, auto_refresh, last_status]
            )
            
            generate_btn.click(
                handle_generation_with_key,
                inputs=[
                    finetune_id_gen, prompt, finetune_strength,
                    endpoint, api_key_gen
                ],
                outputs=image_output
            )
            
            list_btn.click(
                handle_list_finetunes,
                inputs=[api_key_manage],
                outputs=[finetunes_output]
            )
            
            delete_btn.click(
                handle_delete_finetune,
                inputs=[delete_id, api_key_manage],
                outputs=[manage_output]
            ).success(
                handle_list_finetunes,
                inputs=[api_key_manage],
                outputs=[finetunes_output]
            )
            
            # Auto-refresh for manage tab
            self.demo.load(
                auto_refresh_list,
                inputs=[api_key_manage],
                outputs=finetunes_output,
                every=30,
                show_progress=False
            )

    def _get_api(self, api_key: str) -> FluxAPI:
        """Get or create an API client instance."""
        if not self.api or self.api.api_key != api_key:
            self.api = FluxAPI(api_key)
        return self.api

    def _handle_training(
        self,
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
    ) -> str:
        """Handle the training button click event."""
        try:
            # Validate inputs
            if not files:
                return "Error: Please upload at least one image"
            if len(files) > 20:
                return "Error: Maximum 20 images allowed"
            if not finetune_comment:
                return "Error: Please provide a description"
            
            # Create API client
            api = self._get_api(api_key)
            
            # Create ZIP file
            try:
                zip_data, total_size = create_zip_from_files(files)
                logger.info(f"Created ZIP file: {total_size/1024/1024:.2f}MB")
            except Exception as e:
                logger.error(f"Failed to create ZIP file: {str(e)}")
                return f"Error preparing files: {str(e)}"
            
            # Start training
            response = api.request_finetuning(
                file_data=zip_data,
                finetune_comment=finetune_comment,
                trigger_word=trigger_word,
                mode=mode,
                iterations=iterations,
                learning_rate=learning_rate,
                captioning=captioning,
                priority=priority,
                finetune_type=finetune_type,
                lora_rank=lora_rank
            )
            
            return (
                f"Training started!\n"
                f"Finetune ID: {response['id']}\n"
                f"Switch to the Progress tab to monitor status."
            )
            
        except FluxAPIError as e:
            logger.error(f"API error during training: {str(e)}")
            return f"API Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error during training: {str(e)}")
            return f"Error: {str(e)}"

    def _handle_generation(
        self,
        finetune_id: str,
        prompt: str,
        finetune_strength: float,
        endpoint: str,
        api_key: str
    ) -> Union[str, Image.Image]:
        """Handle the generate button click event."""
        try:
            # Validate inputs
            if not finetune_id:
                return "Error: Please enter a Finetune ID"
            if not prompt:
                return "Error: Please enter a prompt"
            
            api = self._get_api(api_key)
            result = api.generate_image(
                finetune_id=finetune_id,
                prompt=prompt,
                finetune_strength=finetune_strength,
                endpoint=endpoint
            )
            
            return result["sample"]
            
        except FluxAPIError as e:
            logger.error(f"API error generating image: {str(e)}")
            return f"API Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error generating image: {str(e)}")
            return f"Error: {str(e)}"

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        logger.info("Launching UI")
        self.demo.launch(**kwargs) 