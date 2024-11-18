import chromadb
import ollama
import json
from typing import List, Dict
from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    OLLAMA_BASE_URL,
    MODEL_NAME,
    MAX_QUERY_RESULTS
)

def get_relevant_context(query: str, n_results: int = MAX_QUERY_RESULTS) -> str:
    """Get relevant context from ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Combine all relevant documents into context
    context = []
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        source = metadata['source']
        context.append(f"From {source}:\n{text}\n")
    
    return "\n".join(context)

def generate_prompt(query: str, context: str) -> str:
    """Generate a prompt that includes context and query."""
    return f"""You are a cybersecurity expert assistant. Use the following context from security documentation to answer the question. 
If the context doesn't contain relevant information, say so and provide general security best practices.

Context:
{context}

Question: {query}

Answer: """

def chat_with_security_expert():
    # Configure Ollama client
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
    
    print("\nSecurity Knowledge Base RAG Chat Interface")
    print("========================================")
    print(f"Using model: {MODEL_NAME}")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("\nYour security question: ")
        
        if query.lower() == 'exit':
            break
        
        print("\nSearching knowledge base...")
        context = get_relevant_context(query)
        
        print("Generating response with Nemotron...")
        prompt = generate_prompt(query, context)
        
        try:
            # Generate response using Ollama with Nemotron
            response = ollama_client.chat(model=MODEL_NAME, messages=[
                {
                    'role': 'system',
                    'content': 'You are a cybersecurity expert assistant. Provide clear, accurate, and practical security advice.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            print("\nResponse:")
            print("=========")
            print(response['message']['content'])
            
        except Exception as e:
            print(f"\nError generating response: {str(e)}")
            print("Please make sure Ollama is running and Nemotron is installed.")
            print("You can install Nemotron with: ollama pull nemotron")

if __name__ == "__main__":
    chat_with_security_expert()
