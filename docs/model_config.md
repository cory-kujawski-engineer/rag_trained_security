# Model Configuration Documentation

## Overview
This document details the configuration settings for the RAG-trained security model variants.

## Model Variants

### Qwen-7B
```python
{
    "name": "Qwen/Qwen-7B",
    "description": "Base 7B parameter model",
    "parameters": {
        "model_type": "causal_lm",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "num_hidden_layers": 32
    },
    "training": {
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "c_attn",
            "c_proj",
            "w1",
            "w2"
        ]
    },
    "quantization": {
        "bits": 4,
        "quant_type": "nf4",
        "double_quant": true
    }
}
```

### Qwen-14B
```python
{
    "name": "Qwen/Qwen-14B",
    "description": "Enhanced 14B parameter model",
    "parameters": {
        "model_type": "causal_lm",
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "intermediate_size": 13696,
        "num_hidden_layers": 40
    },
    "training": {
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "c_attn",
            "c_proj",
            "w1",
            "w2"
        ]
    },
    "quantization": {
        "bits": 4,
        "quant_type": "nf4",
        "double_quant": true
    }
}
```

### Qwen-7B-Chat
```python
{
    "name": "Qwen/Qwen-7B-Chat",
    "description": "Dialogue-optimized 7B model",
    "parameters": {
        "model_type": "causal_lm",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "num_hidden_layers": 32
    },
    "training": {
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "c_attn",
            "c_proj",
            "w1",
            "w2"
        ]
    },
    "quantization": {
        "bits": 4,
        "quant_type": "nf4",
        "double_quant": true
    }
}
```

## Training Configuration

### LoRA Settings
```python
LORA_CONFIG = {
    "r": 8,                     # LoRA attention dimension
    "lora_alpha": 32,          # Alpha scaling factor
    "lora_dropout": 0.05,      # Dropout probability
    "bias": "none",            # Bias configuration
    "task_type": "CAUSAL_LM"   # Task type
}
```

### Training Arguments
```python
TRAINING_ARGS = {
    "output_dir": "./results",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "optim": "paged_adamw_32bit",
    "save_steps": 25,
    "logging_steps": 25,
    "learning_rate": 2e-4,
    "fp16": True,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "constant"
}
```

## Tokenizer Configuration

### Settings
```python
TOKENIZER_CONFIG = {
    "padding_side": "left",
    "pad_token": "<|extra_204|>",
    "model_max_length": 2048,
    "truncation": True,
    "padding": True
}
```

### Special Tokens
```python
SPECIAL_TOKENS = {
    "pad_token": "<|extra_204|>",
    "eos_token": "
}
