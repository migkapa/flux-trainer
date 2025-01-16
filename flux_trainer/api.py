"""FLUX API interaction module."""

import json
import logging
import os
from typing import Dict, List, Optional, Union
import time

import requests
from requests import Response

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FluxAPIError(Exception):
    """Custom exception for FLUX API errors."""
    def __init__(self, message: str, response: Optional[Response] = None):
        super().__init__(message)
        self.response = response


class FluxAPI:
    """FLUX API client for interacting with the FLUX finetuning service."""

    BASE_URL = "https://api.us1.bfl.ai/v1"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the FLUX API client.

        Args:
            api_key: Optional API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("BFL_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in BFL_API_KEY environment variable")
        logger.info("Initialized FLUX API client")

    def _make_request(
        self, method: str, endpoint: str, **kwargs
    ) -> Dict:
        """Make a request to the FLUX API.

        Args:
            method: HTTP method to use
            endpoint: API endpoint to call
            **kwargs: Additional arguments to pass to requests

        Returns:
            Response from the API as a dictionary

        Raises:
            FluxAPIError: If the API request fails or returns invalid response
        """
        headers = {
            "Content-Type": "application/json",
            "X-Key": self.api_key,
        }
        
        url = f"{self.BASE_URL}/{endpoint}"
        logger.debug(f"Making {method} request to {url}")
        
        try:
            response = requests.request(method, url, headers=headers, **kwargs)
            logger.debug(f"Response status code: {response.status_code}")
            
            # Log request details
            if kwargs.get('json'):
                safe_payload = {**kwargs['json']}
                if 'api_key' in safe_payload:
                    safe_payload['api_key'] = '***'
                if 'file_data' in safe_payload:
                    safe_payload['file_data'] = f"<{len(safe_payload['file_data'])} bytes>"
                logger.debug(f"Request payload: {json.dumps(safe_payload, indent=2)}")
            
            # Parse response
            try:
                response_json = response.json() if response.text else {}
                logger.debug(f"Response content: {json.dumps(response_json, indent=2)}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON response: {response.text}")
                raise FluxAPIError(
                    f"Invalid JSON response: {str(e)}\nResponse text: {response.text}",
                    response
                ) from e
            
            # Check for error responses
            if not response.ok:
                error_msg = response_json.get('error', {}).get('message', response.text)
                logger.error(f"API error response: {error_msg}")
                raise FluxAPIError(f"API error: {error_msg}", response)
            
            return response_json
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nResponse: {e.response.text}"
            logger.error(error_msg)
            raise FluxAPIError(error_msg, getattr(e, 'response', None)) from e

    def request_finetuning(
        self,
        file_data: str,
        finetune_comment: str,
        trigger_word: str = "TOK",
        mode: str = "general",
        iterations: int = 300,
        learning_rate: float = 0.00001,
        captioning: bool = True,
        priority: str = "quality",
        finetune_type: str = "full",
        lora_rank: int = 32,
    ) -> Dict:
        """Request a new finetuning job.

        Args:
            file_data: Base64 encoded ZIP file containing training data
            finetune_comment: Description of the finetuning job
            trigger_word: Word to trigger the model
            mode: Training mode (character, product, style, general)
            iterations: Number of training iterations
            learning_rate: Learning rate for training
            captioning: Whether to enable automatic captioning
            priority: Training priority (speed or quality)
            finetune_type: Type of finetuning (full or lora)
            lora_rank: LoRA rank (16 or 32)

        Returns:
            Response from the API containing the finetuning job ID

        Raises:
            FluxAPIError: If the API request fails or returns invalid response
        """
        logger.info(f"Starting finetuning job: {finetune_comment}")
        logger.debug(f"File data size: {len(file_data)} bytes")
        
        payload = {
            "finetune_comment": finetune_comment,
            "trigger_word": trigger_word,
            "file_data": file_data,
            "iterations": iterations,
            "mode": mode,
            "learning_rate": learning_rate,
            "captioning": captioning,
            "priority": priority,
            "lora_rank": lora_rank,
            "finetune_type": finetune_type,
        }

        try:
            response = self._make_request("POST", "finetune", json=payload)
            
            # Validate response
            if not isinstance(response, dict):
                raise FluxAPIError(f"Invalid response type: {type(response)}")
            
            # Check for finetune ID in both possible fields
            finetune_id = response.get('id') or response.get('finetune_id')
            if not finetune_id:
                raise FluxAPIError(f"No finetune ID in response: {response}")
            
            # Normalize response to use 'id' field
            response['id'] = finetune_id
            
            logger.info(f"Finetuning job created with ID: {finetune_id}")
            return response
            
        except FluxAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to start finetuning job: {str(e)}")
            raise FluxAPIError(f"Failed to start finetuning job: {str(e)}")

    def get_progress(self, finetune_id: str) -> Dict:
        """Get the progress of a finetuning job.

        Args:
            finetune_id: ID of the finetuning job

        Returns:
            Progress information
        """
        logger.info(f"Checking progress for job: {finetune_id}")
        return self._make_request(
            "GET", "get_result", params={"id": finetune_id}
        )

    def list_finetunes(self) -> List[Dict]:
        """Get a list of all finetuning jobs.

        Returns:
            List of finetuning jobs
        """
        logger.info("Fetching list of finetuning jobs")
        return self._make_request("GET", "my_finetunes")

    def get_finetune_details(self, finetune_id: str) -> Dict:
        """Get details about a specific finetuning job.

        Args:
            finetune_id: ID of the finetuning job

        Returns:
            Detailed information about the finetuning job
        """
        logger.info(f"Fetching details for job: {finetune_id}")
        response = self._make_request(
            "GET", "finetune_details", params={"finetune_id": finetune_id}
        )
        
        # Handle different response formats
        if isinstance(response, dict):
            return response
        elif isinstance(response, str):
            # Try to extract useful information from string response
            return {
                "finetune_comment": "N/A",
                "status": response if response else "Unknown",
                "created_at": "N/A"
            }
        else:
            logger.warning(f"Unexpected response format for finetune details: {response}")
            return {
                "finetune_comment": "Error: Invalid response format",
                "status": "Unknown",
                "created_at": "N/A"
            }

    def delete_finetune(self, finetune_id: str) -> Dict:
        """Delete a finetuning job.

        Args:
            finetune_id: ID of the finetuning job

        Returns:
            Response from the API
        """
        logger.info(f"Deleting finetuning job: {finetune_id}")
        return self._make_request(
            "POST", "delete_finetune", json={"finetune_id": finetune_id}
        )

    def generate_image(
        self,
        finetune_id: str,
        prompt: str,
        finetune_strength: float = 1.2,
        endpoint: str = "flux-pro-1.1-ultra-finetuned",
        **kwargs
    ) -> Dict:
        """Generate an image using a finetuned model.

        Args:
            finetune_id: ID of the finetuned model to use
            prompt: Text prompt for image generation
            finetune_strength: Strength of the finetuning effect (0-2)
            endpoint: API endpoint to use
            **kwargs: Additional parameters for the endpoint

        Returns:
            Response containing the generated image
        """
        logger.info(f"Generating image with model: {finetune_id}")
        logger.debug(f"Prompt: {prompt}")
        
        payload = {
            "finetune_id": finetune_id,
            "prompt": prompt,
            "finetune_strength": finetune_strength,
            **kwargs
        }

        try:
            # Step 1: Request image generation
            inference_response = self._make_request("POST", endpoint, json=payload)
            inference_id = inference_response.get('id')
            if not inference_id:
                raise FluxAPIError(f"No inference ID in response: {inference_response}")
                
            logger.info(f"Image generation started with ID: {inference_id}")
            
            # Step 2: Poll for results
            max_attempts = 30
            attempt = 0
            while attempt < max_attempts:
                result = self.get_progress(inference_id)
                if result.get("status") == "Ready":
                    if "result" in result and "sample" in result["result"]:
                        logger.info("Image generation completed successfully")
                        return result["result"]
                    else:
                        raise FluxAPIError(f"Invalid result format: {result}")
                elif result.get("status") in ["Failed", "Error"]:
                    error_msg = f"Image generation failed: {result.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    raise FluxAPIError(error_msg)
                
                attempt += 1
                time.sleep(2)  # Wait 2 seconds between checks
            
            raise FluxAPIError("Image generation timed out")
            
        except FluxAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate image: {str(e)}")
            raise FluxAPIError(f"Failed to generate image: {str(e)}") 