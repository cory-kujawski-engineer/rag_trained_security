import chromadb
import json
import random
from typing import List, Dict
import os
import time

def generate_security_questions() -> List[str]:
    """Generate a list of security-related questions."""
    return [
        "What are the key security considerations for {}?",
        "How do you protect against {} attacks?",
        "What are best practices for {} security?",
        "Explain the security implications of {}.",
        "How would you implement secure {} in a production environment?",
        "What are common vulnerabilities in {}?",
        "How do you detect and prevent {} security breaches?",
        "What security controls should be implemented for {}?",
        "Describe the security architecture for {}.",
        "What are the risks associated with {}?"
    ]

def extract_topics_from_metadata(collection) -> List[str]:
    """Extract security topics from document metadata."""
    # Get all documents in smaller batches
    all_docs = []
    batch_size = 10
    total_batches = 10
    
    print("Extracting security topics...")
    for i in range(total_batches):
        try:
            results = collection.query(
                query_texts=[f"security batch {i}"],
                n_results=batch_size,
                include=['metadatas', 'documents']
            )
            all_docs.extend(results['documents'][0])
            print(f"Processed batch {i+1}/{total_batches}")
            time.sleep(1)  # Add delay between batches
        except Exception as e:
            print(f"Error in batch {i}: {str(e)}")
            continue
    
    # Extract key security topics from document content
    topics = set()
    security_terms = ['security', 'vulnerability', 'attack', 'protect', 'secure',
                     'encryption', 'authentication', 'authorization', 'firewall',
                     'network', 'password', 'access', 'control', 'threat']
    
    for doc in all_docs:
        words = doc.lower().split()
        for i, word in enumerate(words[:-1]):
            if word in security_terms and len(words[i+1]) > 3:
                topics.add(words[i+1])
    
    return list(topics)

def generate_training_sample(context: str, topic: str, question_template: str) -> Dict:
    """Generate a training sample with context and response."""
    question = question_template.format(topic)
    
    system_prompt = """You are a cybersecurity expert. Generate a detailed, technical response to the security question.
Use the provided context but also add your expert knowledge. Format the response in clear sections with examples where appropriate."""
    
    user_prompt = f"""Context from security documentation:
{context}

Question: {question}"""
    
    # This will be used as the expected assistant response format
    assistant_prompt = f"""Based on the provided context and security best practices, here's a detailed analysis of {topic} security:

1. Overview
- Understanding {topic} in a security context
- Key security considerations

2. Common Vulnerabilities and Risks
- Known attack vectors
- Potential security implications

3. Security Best Practices
- Implementation guidelines
- Protection mechanisms
- Security controls

4. Monitoring and Detection
- Security monitoring approach
- Incident detection methods
- Response procedures

5. Implementation Examples
- Secure configuration examples
- Code or architecture samples where applicable

6. Additional Considerations
- Compliance requirements
- Integration with existing security measures
- Future security considerations"""

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt}
        ]
    }

def main():
    print("Generating Fine-tuning Dataset for Qwen 2.5")
    print("==========================================")
    
    # Initialize ChromaDB with CPU provider
    client = chromadb.PersistentClient(path="./")
    collection = client.get_or_create_collection(
        name="security_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Get question templates and topics
    questions = generate_security_questions()
    topics = extract_topics_from_metadata(collection)
    
    # Generate training samples
    training_data = []
    samples_per_topic = 2  # Adjust based on how many samples you want
    
    print(f"\nFound {len(topics)} security topics")
    print("Generating training samples...")
    
    for topic in topics:
        print(f"Processing topic: {topic}")
        
        try:
            # Get relevant context for the topic
            results = collection.query(
                query_texts=[topic],
                n_results=3
            )
            
            context = "\n".join(results['documents'][0])
            
            # Generate samples for this topic
            for _ in range(samples_per_topic):
                question_template = random.choice(questions)
                sample = generate_training_sample(context, topic, question_template)
                training_data.append(sample)
                
        except Exception as e:
            print(f"Error processing topic {topic}: {str(e)}")
            continue
    
    # Save training data
    output_file = "security_training_data.json"
    with open(output_file, "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\nGenerated {len(training_data)} training samples")
    print(f"Training data saved to: {output_file}")
    print("\nTo fine-tune Qwen 2.5 with this data:")
    print("1. Convert the JSON to Qwen's expected format")
    print("2. Use: ollama create qwen-security -f ./Modelfile")
    print("3. Fine-tune: ollama train qwen-security security_training_data.json")

if __name__ == "__main__":
    main()
