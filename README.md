# FLUX Trainer 🎨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

A intuitive Gradio UI for Black Forest Labs' FLUX Pro Finetuning API. Train custom image generation models with just a few example images. Perfect for product shots, character designs, style transfer, and more.

> 🚀 **[Try FLUX Pro API](https://docs.bfl.ml/finetuning/)** | 📚 **[Documentation](https://docs.bfl.ml/finetuning/)** | 💬 **[Discussions](https://github.com/migkapa/flux-trainer/discussions)**

## Interface Screenshots

### Training Tab
![Training Interface](assets/4.png)
*Configure and start your training jobs with an intuitive interface*

### Progress Monitoring
![Progress Monitoring](assets/3.png)
*Track your training progress in real-time*

### Model Management
![Model Management](assets/2.png)
*Manage your finetuned models and view their details*

### Image Generation
![Image Generation](assets/1.png)
*Generate images using your custom-trained models*

## ✨ Features

- 🖼️ **Intuitive UI**: Drag-and-drop interface for training data management
- ⚡ **Flexible Training**: Quick exploration (100 iterations) to deep training (1000+ iterations)
- 🎯 **Multiple Modes**: 
  - `character`: Consistent character generation
  - `product`: Product visualization and marketing
  - `style`: Artistic style transfer
  - `general`: Broader concept training
- 🔄 **Real-time Monitoring**: Live training progress and error tracking
- 🎨 **Full Integration**: Works with all FLUX models (Pro, Ultra, Raw)
- 🚀 **High Resolution**: Support for outputs up to 4 megapixels

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- Poetry
- FLUX API key ([Get one here](https://docs.bfl.ml/finetuning/))

### Installation

```bash
# Clone the repo
git clone https://github.com/migkapa/flux-trainer.git
cd flux-trainer

# Install dependencies
poetry install

# Set up environment
echo "BFL_API_KEY=your_api_key_here" > .env
```

### Running the UI

```bash
poetry run python -m flux_trainer
```

Navigate to `http://localhost:7860` in your browser.

## 🎓 Advanced Usage

### Training Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| `iterations` | Training steps | 300 | 100-1000 |
| `learning_rate` | Training precision | 0.00001 | 0.00001-0.0001 |
| `priority` | Training mode | "quality" | "quality"/"speed" |
| `finetune_type` | Training approach | "full" | "full"/"lora" |
| `lora_rank` | LoRA complexity | 32 | 16/32 |

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `BFL_API_KEY` | Your FLUX API key | Yes |
| `FLUX_DEFAULT_DESCRIPTION` | Default training description | No |

## 🤝 Contributing

Contributions are welcome! Check out our [Contributing Guidelines](CONTRIBUTING.md).

```bash
# Development setup
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .
```

## 📝 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Black Forest Labs](https://docs.bfl.ml/finetuning/) - For the amazing FLUX Pro API
- [Gradio](https://gradio.app/) - For the UI framework

---

<p align="center">
  <sub>Built with ❤️ by the community</sub>
</p> 