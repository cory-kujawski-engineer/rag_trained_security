import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ChromaDB Configuration
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './')
CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'security_knowledge_base')

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
MODEL_NAME = os.getenv('MODEL_NAME', 'qwen')

# Security Settings
MAX_QUERY_RESULTS = int(os.getenv('MAX_QUERY_RESULTS', '5'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
