import chromadb
import ollama
import os
from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Create or get collection
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# Function to upsert text chunks into ChromaDB with metadata
def upsert_into_chromadb(chunks, metadata):
    # Upsert documents into ChromaDB with metadata and unique IDs
    collection.add(
        documents=chunks,
        metadatas=metadata,
        ids=[f"chunk_{i+1}" for i in range(len(chunks))]  # Generating unique ids like "chunk_1", "chunk_2", etc.
    )

# Query ChromaDB
def query_chromadb(prompt, n_results=3):
    # ChromaDB will generate the embedding for the query and find the most similar chunks
    results = collection.query(
        query_texts=[prompt],
        n_results=n_results
    )
    return results

def read_security_file(filepath):
    """Read a security text file and return its contents."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try with a different encoding if utf-8 fails
        try:
            with open(filepath, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file {filepath}: {str(e)}")
            return None
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    if not text:
        return chunks
    
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Find the last period or newline in the chunk to avoid cutting sentences
        if end < len(text):
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            end = max(last_period, last_newline) if max(last_period, last_newline) > start else end
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        start = end - overlap
    return chunks

def ingest_security_files():
    """Ingest all security text files from /usr/share/security/"""
    security_dir = "/usr/share/security"
    try:
        files = [f for f in os.listdir(security_dir) if f.endswith('.txt')]
        total_files = len(files)
        print(f"\nStarting ingestion of {total_files} files...")
        processed = 0
        
        for filename in files:
            processed += 1
            filepath = os.path.join(security_dir, filename)
            print(f"\nProcessing {filename}... ({processed}/{total_files}) - {(processed/total_files)*100:.1f}%")
            
            # Read file content
            content = read_security_file(filepath)
            if content is None:
                print(f"Skipping {filename} - Could not read file")
                continue
            
            # Chunk the content
            chunks = chunk_text(content)
            if not chunks:
                print(f"Skipping {filename} - No valid chunks generated")
                continue
            
            print(f"Generated {len(chunks)} chunks from {filename}")
            
            # Prepare metadata
            metadata = [{
                "source": filename,
                "type": "security_text",
                "chunk_index": i,
                "total_chunks": len(chunks)
            } for i in range(len(chunks))]
            
            # Upsert into ChromaDB
            try:
                upsert_into_chromadb(chunks, metadata)
                print(f"Successfully ingested {filename}")
            except Exception as e:
                print(f"Error ingesting {filename}: {str(e)}")
            
            if processed % 10 == 0:
                print(f"\nProgress: {processed}/{total_files} files processed ({(processed/total_files)*100:.1f}%)")
        
        print(f"\nIngestion complete! Processed {processed}/{total_files} files")
                
    except Exception as e:
        print(f"Error accessing directory {security_dir}: {str(e)}")

def main():
    """Main function to run the ingestion process."""
    print("Starting security text ingestion...")
    print("This will ingest all .txt files from /usr/share/security/ into ChromaDB")
    ingest_security_files()
    print("Ingestion complete!")

if __name__ == '__main__':
    main()
