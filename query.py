import chromadb
import json
from typing import List, Dict

def format_results(results: Dict, max_length: int = 300) -> str:
    """Format the results in a readable way."""
    formatted_output = []
    
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        # Truncate text if it's too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        result = f"\n--- Result {i+1} (Distance: {distance:.4f}) ---"
        result += f"\nSource: {metadata['source']}"
        result += f"\nChunk {metadata['chunk_index'] + 1} of {metadata['total_chunks']}"
        result += f"\nText: {text}\n"
        
        formatted_output.append(result)
    
    return "\n".join(formatted_output)

def query_knowledge_base(query: str, n_results: int = 3) -> str:
    """Query the security knowledge base."""
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./")
    collection = client.get_or_create_collection(name="security_knowledge_base")
    
    # Perform the query
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    return format_results(results)

def main():
    print("\nSecurity Knowledge Base Query Interface")
    print("======================================")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("\nEnter your security-related question: ")
        
        if query.lower() == 'exit':
            break
        
        try:
            n_results = int(input("How many results would you like? (default: 3): ") or "3")
        except ValueError:
            n_results = 3
        
        print("\nSearching knowledge base...")
        results = query_knowledge_base(query, n_results)
        print(results)

if __name__ == "__main__":
    main()
