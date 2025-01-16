FLUX Finetuning Beta Guide
Overview
The BFL Finetuning API enables you to customize FLUX Pro and FLUX Ultra using 1 - 20 images of your own visual content, and optionally, text descriptions.

Getting Started: Step-by-Step Guide
Prepare Your Images

Create a local folder containing your training images
Supported formats: JPG, JPEG, PNG, and WebP
Recommended are more than 5 images
Note

High-quality datasets with clear, articulated subjects/objects/styles significantly improve training results. Higher resolution source images help but are capped at 1MP.

Add Text Descriptions (Optional)

Create text files with descriptions for your images
Text files should share the same name as their corresponding images
Example: if your image is "sample.jpg", create "sample.txt"
Package Your Data

Compress your folder into a ZIP file
Configure Training Parameters

Select appropriate hyperparameters (see detailed parameters section below)
Submit Training Task

Use the provided Python script to submit your finetuning task
Run Inference

Once training is complete, use your model through the available endpoints
Training Parameters
Required Parameters
mode
  Purpose: Determines the finetuning approach based on your concept
  Options: "character", "product", "style", "general"
Note

In "general" mode, the entire image is captioned when captioning is True without specific focus areas. No subject specific improvements will be made.

finetune_comment
 Purpose: Descriptive note to identify your fine-tune since names are UUIDs. Will be displayed in finetune_details.
Optional Parameters
iterations
 Minimum: 100
 Default: 300
 Purpose: Defines training duration

learning_rate
  Default: 0.00001

Note

Lower values may be needed for certain scenarios Warning: Higher values might destabilize training
priority
 Options: "speed", "quality"
 The speed priority will improve training and inference speed
 Default: "quality"

captioning
  Type: Boolean
  Default: True
  Purpose: Enables/disables automatic image captioning

trigger_word
  Default: "TOK"
  Purpose: Unique word/phrase that will be used in the captions, to reference the newly introduced concepts

lora_rank
  Default: 32
  Choose between 32 and 16. A lora_rank of 16 can increase training efficiency and decrease loading times.

finetune_type
  Default: "full"
  Choose between “full” for a full finetuning + post hoc extraction of the trained weights into a LoRA or “lora” for a raw LoRA training

Inference Endpoints
Available endpoints for your finetuned model:

/flux-pro-1.1-ultra-finetuned
/flux-pro-finetuned
/flux-pro-1.0-depth-finetuned
/flux-pro-1.0-canny-finetuned
/flux-pro-1.0-fill-finetuned
Additional inference parameters:
Note

The endpoints have additionally all input parameters that their non finetuned sibling endpoints have.

finetune_id:
  References your specific model

Note

find the finetune_id either in my_finetunes or in the return dict of your /finetune POST

finetune_strength:
  Value range: 0-2
  Controls finetune influence
  Increase this value if your target concept isn't showing up strongly enough. The optimal setting depends on your finetune and prompt.

Implementation Guide
Example Python Implementation


python

"""
bfl_finetune.py
Example code for using the BFL finetuning API.

Preparation:

Set your BFL API key:

export BFL_API_KEY=<your api key>

Install requirements:

pip install requests fire

Assuming you have prepared your images in a `finetuning.zip` file:
# submit finetuning task
$ python bfl_finetune.py request_finetuning finetuning.zip myfirstfinetune
id:            <finetune_id>

# query status
$ python bfl_finetune.py finetune_progress <finetune_id>
id:       <finetune_id>
status:   Pending
result:   null
progress: null

# once status shows Ready, run inference (defaults to flux-pro-1.1-ultra-finetuned)
$ python bfl_finetune.py finetune_inference <finetune_id> --prompt="image of a TOK"
finetune_id: <inference_id>

# retrieve inference result
$ python bfl_finetune.py get_inference <inference_id>
id:       <inference_id>
status:   Ready
result:   {"sample": <result_url>, "prompt": "image of a TOK"}
progress: null
"""

import os
import base64
import requests


def request_finetuning(
    zip_path,
    finetune_comment,
    trigger_word="TOK",
    mode="general",
    api_key=None,
    iterations=300,
    learning_rate=0.00001,
    captioning=True,
    priority="quality",
    finetune_type="full",
    lora_rank=32,
):
    """
    Request a finetuning using the provided ZIP file.

    Args:
        zip_path (str): Path to the ZIP file containing training data
        finetune_comment (str): Comment for the finetune_details
        trigger_word (str): Trigger word for the model
        mode (str): Mode for caption generation
        api_key (str): API key for authentication
        iterations (int): Number of training iterations
        learning_rate (float): Learning rate for optimization
        captioning (bool): Enable/disable auto-captioning
        priority (str): Training quality setting
        lora_rank (str): Lora rank
        Finetune_type (str): "full" or "lora"

    Returns:
        dict: API response

    Raises:
        FileNotFoundError: If ZIP file is missing
        requests.exceptions.RequestException: If API request fails
    """
    if api_key is None:
        if "BFL_API_KEY" not in os.environ:
            raise ValueError(
                "Provide your API key via --api_key or an environment variable BFL_API_KEY"
            )
        api_key = os.environ["BFL_API_KEY"]

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found at {zip_path}")

    assert mode in ["character", "product", "style", "general"]

    with open(zip_path, "rb") as file:
        encoded_zip = base64.b64encode(file.read()).decode("utf-8")

    url = "https://api.us1.bfl.ai/v1/finetune"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }
    payload = {
        "finetune_comment": finetune_comment,
        "trigger_word": trigger_word,
        "file_data": encoded_zip,
        "iterations": iterations,
        "mode": mode,
        "learning_rate": learning_rate,
        "captioning": captioning,
        "priority": priority,
        "lora_rank": lora_rank,
        "finetune_type": finetune_type,
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Finetune request failed:\n{str(e)}\n{response.content.decode()}"
        )


def finetune_progress(
    finetune_id,
    api_key=None,
):
    if api_key is None:
        if "BFL_API_KEY" not in os.environ:
            raise ValueError(
                "Provide your API key via --api_key or an environment variable BFL_API_KEY"
            )
        api_key = os.environ["BFL_API_KEY"]
    url = "https://api.us1.bfl.ai/v1/get_result"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }
    payload = {
        "id": finetune_id,
    }

    response = requests.get(url, headers=headers, params=payload)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Finetune progress failed:\n{str(e)}\n{response.content.decode()}"
        )


def finetune_list(
    api_key=None,
):
    if api_key is None:
        if "BFL_API_KEY" not in os.environ:
            raise ValueError(
                "Provide your API key via --api_key or an environment variable BFL_API_KEY"
            )
        api_key = os.environ["BFL_API_KEY"]
    url = "https://api.us1.bfl.ai/v1/my_finetunes"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }

    response = requests.get(url, headers=headers)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Finetune listing failed:\n{str(e)}\n{response.content.decode()}"
        )


def finetune_details(
    finetune_id,
    api_key=None,
):
    if api_key is None:
        if "BFL_API_KEY" not in os.environ:
            raise ValueError(
                "Provide your API key via --api_key or an environment variable BFL_API_KEY"
            )
        api_key = os.environ["BFL_API_KEY"]
    url = "https://api.us1.bfl.ai/v1/finetune_details"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }
    payload = {
        "finetune_id": finetune_id,
    }

    response = requests.get(url, headers=headers, params=payload)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Finetune details failed:\n{str(e)}\n{response.content.decode()}"
        )


def finetune_delete(
    finetune_id,
    api_key=None,
):
    if api_key is None:
        if "BFL_API_KEY" not in os.environ:
            raise ValueError(
                "Provide your API key via --api_key or an environment variable BFL_API_KEY"
            )
        api_key = os.environ["BFL_API_KEY"]

    url = "https://api.us1.bfl.ai/v1/delete_finetune"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }
    payload = {
        "finetune_id": finetune_id,
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Finetune deletion failed:\n{str(e)}\n{response.content.decode()}"
        )


def finetune_inference(
    finetune_id,
    finetune_strength=1.2,
    endpoint="flux-pro-1.1-ultra-finetuned",
    api_key=None,
    **kwargs,
):
    if api_key is None:
        if "BFL_API_KEY" not in os.environ:
            raise ValueError(
                "Provide your API key via --api_key or an environment variable BFL_API_KEY"
            )
        api_key = os.environ["BFL_API_KEY"]

    url = f"https://api.us1.bfl.ai/v1/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }
    payload = {
        "finetune_id": finetune_id,
        "finetune_strength": finetune_strength,
        **kwargs,
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Finetune inference failed:\n{str(e)}\n{response.content.decode()}"
        )


def get_inference(
    id,
    api_key=None,
):
    if api_key is None:
        if "BFL_API_KEY" not in os.environ:
            raise ValueError(
                "Provide your API key via --api_key or an environment variable BFL_API_KEY"
            )
        api_key = os.environ["BFL_API_KEY"]
    url = "https://api.us1.bfl.ai/v1/get_result"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }
    payload = {
        "id": id,
    }

    response = requests.get(url, headers=headers, params=payload)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Inference retrieval failed:\n{str(e)}\n{response.content.decode()}"
        )


if __name__ == "__main__":
    import fire

    fire.Fire()

Best Practices and Tips
Enhancing Concept Representation

Try finetune_strength values up to ~1.4 if the concept is not present
Try a lower finetune_strength if the image has visible artifacts
Style training may benefit from even higher values
Character Training

Avoid multiple characters in single images
Use manual captions when multiple characters are unavoidable
Consider disabling auto-captioning in complex scenes with many characters
Quality Considerations

Use high-quality training images
Adjust learning rate based on training stability
Monitor training progress and adjust parameters as needed