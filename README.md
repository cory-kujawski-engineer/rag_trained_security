# RAG-Trained Security Model Project

## Overview
This project implements a security-focused language model using Retrieval-Augmented Generation (RAG) techniques. The system is built around the Qwen model family and is specifically designed for security-related tasks and analysis.

## Project Structure
```
rag_trained_security/
├── model_training/           # Model fine-tuning and training scripts
│   ├── finetune_qwen.py     # Main fine-tuning script for Qwen models
│   └── README.md            # Training-specific documentation
├── data_preparation/        # Data processing and preparation scripts
│   └── README.md           # Data preparation documentation
├── utils/                  # Utility functions and helper scripts
│   └── README.md          # Utilities documentation
└── docs/                  # Detailed documentation
    ├── training.md        # Training process documentation
    ├── data_format.md     # Data format specifications
    └── model_config.md    # Model configuration details
```

## Features
- Support for multiple Qwen model variants (7B, 14B, Chat)
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- 4-bit quantization for efficient training
- Flexible data input format
- Comprehensive logging and error handling

## Supported Models
1. **Qwen-7B Base**
   - Standard 7B parameter model
   - Optimal for general security tasks
   - Balanced performance and resource usage

2. **Qwen-14B**
   - Larger 14B parameter model
   - Enhanced reasoning capabilities
   - Suitable for complex security analysis

3. **Qwen-7B Chat**
   - Conversation-optimized 7B model
   - Ideal for interactive security applications
   - Better response formatting

## Quick Start
1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your configurations
   nano .env
   ```
   
   Key environment variables:
   - `CHROMA_DB_PATH`: Path to ChromaDB storage
   - `CHROMA_COLLECTION_NAME`: Name of the ChromaDB collection
   - `OLLAMA_BASE_URL`: Ollama API endpoint
   - `MODEL_NAME`: Name of the model to use
   - `MAX_QUERY_RESULTS`: Number of results to return per query
   - `CHUNK_SIZE`: Size of text chunks for processing
   - `CHUNK_OVERLAP`: Overlap between text chunks

3. **List Available Models**
   ```bash
   python model_training/finetune_qwen.py --list-models
   ```

4. **Train Model**
   ```bash
   python model_training/finetune_qwen.py --model qwen-7b --training-file your_data.json
   ```

## Documentation
- [Training Process](docs/training.md): Detailed guide on model training
- [Data Format](docs/data_format.md): Specifications for training data
- [Model Configuration](docs/model_config.md): Model-specific settings

## System Requirements
- NVIDIA GPU with 24GB+ VRAM (RTX 4090 or better)
- CUDA 12.4+
- 64GB+ System RAM
- Python 3.10+

## Training Data
The system expects training data in JSON format with specific fields for security-related content. See [Data Format](docs/data_format.md) for details.

## Model Configuration
Each model variant has specific configurations for:
- LoRA parameters (rank, alpha, dropout)
- Target modules for fine-tuning
- Quantization settings
- Padding and tokenization

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## License
[Specify your license here]

## Contact
[Your contact information]

## Acknowledgments
- Qwen model team
- PEFT library contributors
- Hugging Face team
