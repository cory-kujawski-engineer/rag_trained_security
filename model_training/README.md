# Model Training Suite

## Overview
This directory contains scripts and configurations for fine-tuning Qwen models on security-related tasks using Parameter-Efficient Fine-Tuning (PEFT) techniques.

## Files
- `finetune_qwen.py`: Main training script with support for multiple Qwen models

## Usage

### Command-line Arguments
```bash
python finetune_qwen.py [OPTIONS]
```

Options:
- `--model`: Model to fine-tune (e.g., qwen-7b, qwen-14b, qwen-7b-chat)
- `--list-models`: List available models and exit
- `--training-file`: Path to training data file (default: training_data.json)

### Interactive Mode
If no model is specified via command line, the script runs in interactive mode and prompts for model selection.

## Model Configurations

### Qwen-7B
```python
{
    "name": "Qwen/Qwen-7B",
    "description": "Qwen 7B base model",
    "target_modules": ["c_attn", "c_proj", "w1", "w2"],
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05
}
```

### Qwen-14B
```python
{
    "name": "Qwen/Qwen-14B",
    "description": "Qwen 14B base model",
    "target_modules": ["c_attn", "c_proj", "w1", "w2"],
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05
}
```

### Qwen-7B-Chat
```python
{
    "name": "Qwen/Qwen-7B-Chat",
    "description": "Qwen 7B chat model",
    "target_modules": ["c_attn", "c_proj", "w1", "w2"],
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05
}
```

## Training Process

1. **Model Initialization**
   - Load model with 4-bit quantization
   - Configure tokenizer with proper padding
   - Enable gradient checkpointing

2. **Data Processing**
   - Load and validate training data
   - Apply tokenization with left-side padding
   - Prepare dataset for training

3. **Fine-tuning Configuration**
   - Set up LoRA adapters
   - Configure training arguments
   - Initialize trainer

4. **Training Loop**
   - Run training with specified epochs
   - Monitor GPU memory usage
   - Save checkpoints

5. **Model Saving**
   - Save trained model
   - Save LoRA weights
   - Export configuration

## Error Handling
The script includes comprehensive error handling for:
- GPU memory issues
- Data format problems
- Model loading failures
- Training interruptions

## Logging
Detailed logging is implemented for:
- Training progress
- GPU memory usage
- Model configuration
- Error messages

## Requirements
See main `requirements.txt` for detailed dependencies.

## Contributing
1. Follow PEP 8 style guide
2. Add comprehensive docstrings
3. Include error handling
4. Update documentation
