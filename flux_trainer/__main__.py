"""Main entry point for the FLUX Trainer application."""

import os
from pathlib import Path

from dotenv import load_dotenv

from flux_trainer.ui import FluxTrainerUI


def main():
    """Run the FLUX Trainer application."""
    # Load environment variables from .env files
    # First try local .env
    local_env = Path(".env")
    if local_env.exists():
        load_dotenv(local_env)
        print("Loaded API key from local .env file")
    
    # Then try global .env (will not override existing values)
    global_env = Path.home() / ".flux" / ".env"
    if global_env.exists():
        load_dotenv(global_env, override=False)
        print("Loaded API key from global .env file")
    
    # Get port from environment variable or use a range
    port = os.getenv("GRADIO_SERVER_PORT", None)
    if port:
        port = int(port)
    
    # Create the UI and launch it
    ui = FluxTrainerUI()
    ui.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=port,  # Use environment variable or let Gradio find an available port
        share=False,  # Don't create a public URL
    )


if __name__ == "__main__":
    main() 