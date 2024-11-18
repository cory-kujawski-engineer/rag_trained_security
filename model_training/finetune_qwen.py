import os
import json
import torch
import logging
import argparse
import psutil
import GPUtil
import requests
from typing import Dict, Any, List
from requests.exceptions import RequestException
from functools import partial

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info("Logging initialized successfully")

# Add supported models
SUPPORTED_MODELS = {
    "qwen-7b": {
        "name": "Qwen/Qwen-7B",
        "description": "Qwen 7B base model",
        "target_modules": ["c_attn", "c_proj", "w1", "w2"],
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05
    },
    "qwen-14b": {
        "name": "Qwen/Qwen-14B",
        "description": "Qwen 14B base model",
        "target_modules": ["c_attn", "c_proj", "w1", "w2"],
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05
    },
    "qwen-7b-chat": {
        "name": "Qwen/Qwen-7B-Chat",
        "description": "Qwen 7B chat model",
        "target_modules": ["c_attn", "c_proj", "w1", "w2"],
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05
    }
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen models for security tasks")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to fine-tune (e.g., qwen-7b)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--training-file",
        type=str,
        default="training_data.json",
        help="Path to training data file (default: training_data.json)"
    )
    return parser.parse_args()

def check_gpu():
    """Check GPU availability and configuration."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Please check your PyTorch installation.")
        return False
    
    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} CUDA device(s)")
    
    # Get the current GPU device
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    logger.info(f"Using GPU: {device_name} (Device {current_device})")
    
    # Get memory information
    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    total_memory_gb = total_memory / (1024**3)
    logger.info(f"Total GPU memory: {total_memory_gb:.2f} GB")
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    logger.info(f"CUDA Version: {cuda_version}")
    
    return True

def get_gpu_memory_info():
    """Get GPU memory information."""
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logger.info(f"\nGPU {gpu.id}: {gpu.name}")
            logger.info(f"Memory Total: {gpu.memoryTotal} MB")
            logger.info(f"Memory Free: {gpu.memoryFree} MB")
            logger.info(f"Memory Used: {gpu.memoryUsed} MB")
            logger.info(f"Memory Utilization: {gpu.memoryUtil*100:.2f}%")
        return gpus[0].memoryTotal if gpus else 0
    except Exception as e:
        logger.error(f"Error getting GPU info: {str(e)}")
        return 0

def estimate_model_size():
    """Estimate model memory requirements."""
    try:
        # Base model size (Qwen-7B)
        base_model_params = 7 * 1024 * 1024 * 1024  # 7B parameters
        
        # 4-bit quantization reduces memory by ~4x
        quantized_model_size = (base_model_params * 4) // 8  # 4 bits per parameter
        
        # LoRA adapters are much smaller
        lora_size = 32 * 1024 * 1024  # Rough estimate for LoRA layers
        
        # Additional memory for gradients, optimizer states, etc.
        overhead_factor = 1.2
        
        total_estimated = (quantized_model_size + lora_size) * overhead_factor
        
        return {
            'base_model_gb': base_model_params * 4 / (1024**3),  # Full precision size
            'quantized_gb': quantized_model_size / (1024**3),    # 4-bit quantized size
            'lora_mb': lora_size / (1024**2),                    # LoRA adapters size
            'total_gb': total_estimated / (1024**3)              # Total estimated
        }
    except Exception as e:
        logger.error(f"Error estimating model size: {str(e)}")
        return None

def check_system_requirements():
    """Check if system meets requirements for training."""
    logger.info("\nChecking system requirements...")
    
    # Check GPU first
    if not check_gpu():
        return False
    
    # Get available GPU memory
    gpu_memory = get_gpu_memory_info()
    if gpu_memory < 24000:  # Minimum 24GB recommended for Qwen-7B with 4-bit quantization
        logger.warning(f"GPU memory ({gpu_memory}MB) might be insufficient for training Qwen-7B")
        logger.warning("Recommended minimum: 24GB")
        return False
    
    # Check CPU memory
    system_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
    logger.info(f"\nSystem Memory: {system_memory:.2f}GB")
    
    if system_memory < 16:
        logger.warning("Less than 16GB of system memory available")
        logger.warning("This might cause issues during training")
        return False
    
    logger.info("\nSystem meets memory requirements for training!")
    return True

def load_training_data(file_path: str) -> List[Dict]:
    """Load and preprocess the training data."""
    logger.info(f"Loading training data from {file_path}")
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training data file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} training examples")
        return data
    except FileNotFoundError as e:
        logger.error(f"Training data file not found: {file_path}")
        logger.error("Please create a training data file before proceeding.")
        raise
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

def create_dataset(data: List[Dict]) -> Dataset:
    """Create a HuggingFace dataset from our data."""
    logger.info("Creating HuggingFace Dataset")
    try:
        dataset = Dataset.from_list(data)
        logger.info(f"Dataset created successfully with {len(dataset)} examples")
        return dataset
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise

def tokenize_function(examples, tokenizer):
    """
    Tokenize the examples using the provided tokenizer.
    
    Args:
        examples: Dictionary containing the examples to tokenize
        tokenizer: The tokenizer to use
        
    Returns:
        Dictionary containing the tokenized examples
    """
    logger.info(f"Tokenizing batch of {len(examples['text'])} examples")
    
    try:
        # Verify tokenizer configuration
        logger.info(f"Tokenizer config - PAD: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
        
        if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
            logger.error("Padding token not properly configured")
            raise ValueError("Padding token not properly configured")
        
        # Tokenize the text
        encoded = tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            pad_to_multiple_of=8,  # For efficient tensor operations
        )
        
        # For causal language modeling, labels are the same as input_ids
        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["input_ids"].clone(),
        }
        
        logger.info(f"Tokenization successful. Shape: {result['input_ids'].shape}")
        return result
        
    except Exception as e:
        logger.error(f"Error in tokenization: {str(e)}")
        raise

def list_supported_models():
    """List all supported models with their descriptions."""
    logger.info("\nSupported Models:")
    for key, model in SUPPORTED_MODELS.items():
        logger.info(f"- {key}: {model['description']}")
    logger.info("")

def get_model_config(model_key):
    """Get the configuration for a specific model."""
    if model_key not in SUPPORTED_MODELS:
        available_models = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(f"Model '{model_key}' not supported. Available models: {available_models}")
    return SUPPORTED_MODELS[model_key]

def prepare_model_and_tokenizer(model_key: str):
    """Prepare the model and tokenizer."""
    model_config = get_model_config(model_key)
    model_name = model_config["name"]
    logger.info(f"Preparing model and tokenizer for {model_name}")
    
    try:
        # Initialize quantization config
        logger.info("Initializing quantization config...")
        compute_dtype = getattr(torch, "float16")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set up padding configuration
        logger.info("Setting up padding configuration...")
        logger.info(f"Initial tokenizer config - EOS: {tokenizer.eos_token}, PAD: {tokenizer.pad_token}")
        logger.info(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        
        # For Qwen, we need to set up the padding token
        logger.info("Setting up padding token...")
        
        # Get the last token ID in vocabulary to use as padding token
        pad_token_id = len(tokenizer) - 1
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_token_id)
        tokenizer.pad_token_id = pad_token_id
        
        # Set left padding for Qwen
        tokenizer.padding_side = "left"
        
        # Verify configuration
        logger.info(f"Final tokenizer config - PAD: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for training
        logger.info("Preparing model for training...")
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        logger.info("Configuring LoRA...")
        target_modules = model_config["target_modules"]
        logger.info(f"Using target modules: {target_modules}")
        
        config = LoraConfig(
            r=model_config["lora_r"],
            lora_alpha=model_config["lora_alpha"],
            target_modules=target_modules,
            lora_dropout=model_config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model
        logger.info("Applying LoRA to model...")
        model = get_peft_model(model, config)
        
        logger.info("Model and tokenizer preparation completed successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error in model preparation: {str(e)}")
        raise

def main():
    """Main function to run the fine-tuning process."""
    args = parse_arguments()
    
    if args.list_models:
        list_supported_models()
        return
    
    try:
        logger.info("Starting fine-tuning process")
        
        # If model is not provided, ask user interactively
        model_key = args.model
        if model_key is None:
            list_supported_models()
            while True:
                model_key = input("Enter the model key to use (e.g., qwen-7b): ").strip().lower()
                if model_key in SUPPORTED_MODELS:
                    break
                logger.error(f"Invalid model key. Please choose from: {', '.join(SUPPORTED_MODELS.keys())}")
        
        logger.info(f"Selected model: {SUPPORTED_MODELS[model_key]['name']}")
        
        # Check for training data file first
        training_file = args.training_file
        if not os.path.exists(training_file):
            logger.error(f"Training data file '{training_file}' not found!")
            logger.error("Please create a training data file before running fine-tuning.")
            return
        
        logger.info("Checking system requirements...")
        if not check_system_requirements():
            logger.error("System requirements not met. Please check the warnings above.")
            return
        
        logger.info("Loading training data...")
        data = load_training_data(training_file)
        
        logger.info("Creating dataset...")
        dataset = create_dataset(data)
        
        logger.info("Preparing model and tokenizer...")
        model, tokenizer = prepare_model_and_tokenizer(model_key)
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=2e-4,
            fp16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            report_to="none",
            gradient_checkpointing=True,
            bf16=False,
        )
        
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        
        # Log GPU memory before training
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"GPU memory allocated before training: {gpu_memory:.2f} GB")
        
        logger.info("Starting training...")
        try:
            trainer.train()
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            # Log GPU memory after error
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            logger.error(f"GPU memory allocated after error: {gpu_memory:.2f} GB")
            raise
        
        # Log GPU memory after training
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"GPU memory allocated after training: {gpu_memory:.2f} GB")
        
        logger.info("Saving model...")
        trainer.save_model("./final_model")
        logger.info("Model saved successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        raise

if __name__ == "__main__":
    logger.info("Script starting...")
    main()
