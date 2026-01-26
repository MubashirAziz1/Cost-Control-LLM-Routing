"""Simple script to test classifier model with user query."""
from src.services.hf.factory import make_huggingface_client

def main():
    """Take user query and return classifier output."""
    # Get settings and initialize client
    client = make_huggingface_client()
    
    # Get user query
    user_query = "If global temperatures rise 2°C, explain the cascading effects on ocean currents, weather patterns, and agriculture."
    
    # Classify the query
    result = client.classify(user_query)
    
    # Print the result
    print(f"\nClassifier Output: {result}")


if __name__ == "__main__":
    main()
