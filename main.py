"""Simple script to test classifier model with user query."""
from src.services.hf.factory import make_huggingface_client

def main():
    """Take user query and return classifier output."""
    # Get settings and initialize client
    client = make_huggingface_client()
    
    # Get user query
    user_query = "What is the capital of Pakistan?"
    
    # Classify the query
    result = client.classify(user_query)
    
    # Print the result
    print(f"\nClassifier Output: {result}")


if __name__ == "__main__":
    main()
