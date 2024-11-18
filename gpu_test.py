import chromadb
import time
import statistics

def run_performance_test(n_queries=100):
    print(f"\nRunning performance test with {n_queries} queries...")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./")
    collection = client.get_or_create_collection(name="security_knowledge_base")
    
    # Test queries
    test_queries = [
        "What are common security vulnerabilities?",
        "How to protect against SQL injection?",
        "Best practices for password storage?",
        "Network security measures?",
        "Data encryption methods?"
    ]
    
    # Run queries and measure time
    query_times = []
    
    print("\nStarting queries...")
    for i in range(n_queries):
        query = test_queries[i % len(test_queries)]
        
        start_time = time.time()
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        end_time = time.time()
        
        query_time = (end_time - start_time) * 1000  # Convert to milliseconds
        query_times.append(query_time)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_queries} queries")
    
    # Calculate statistics
    avg_time = statistics.mean(query_times)
    median_time = statistics.median(query_times)
    min_time = min(query_times)
    max_time = max(query_times)
    
    print("\nPerformance Results:")
    print(f"Average query time: {avg_time:.2f} ms")
    print(f"Median query time: {median_time:.2f} ms")
    print(f"Fastest query: {min_time:.2f} ms")
    print(f"Slowest query: {max_time:.2f} ms")
    print(f"Total queries: {len(query_times)}")

if __name__ == "__main__":
    run_performance_test()
