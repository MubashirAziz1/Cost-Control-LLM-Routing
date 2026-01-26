"""Simple script to test classifier model with user query and generate response."""
from src.services.hf.factory import make_huggingface_client
from src.services.groq.factory import make_groq_client

def main():
    """Take user query, classify it, and pass to the selected model for generation."""
    # Initialize clients
    hf_client = make_huggingface_client()
    groq_client = make_groq_client()
    
    # Get user query
    user_query = "If I have 3 apples and buy 5 more, then give away 2, how many do I have?"
    
    # Step 1: Classify the query
    classification_result = hf_client.classify(user_query)

    print(classification_result)
    
    # Parse the classification result (it returns text, need to extract difficulty level)
    classification_lower = classification_result.lower().strip()
    if "simple" in classification_lower:
        difficulty = "simple"
    elif "medium" in classification_lower:
        difficulty = "medium"
    elif "complex" in classification_lower:
        difficulty = "complex"
    
    print(f"\nClassification Result: {classification_result}")
    print(f"Parsed Difficulty: {difficulty}")
    
    # Step 2: Pass query to the selected model based on classification
    if difficulty == "simple":
        response = hf_client.easy_task(user_query)
        model_name = "HF Model (Simple)"
    elif difficulty == "medium":
        response = groq_client.medium_task(user_query)
        model_name = "Groq Model (Medium)"
    else:  # complex
        # For complex tasks, use Groq as well (or you can add a complex_task method)
        response = groq_client.medium_task(user_query)
        model_name = "Groq Model (Complex)"
    
    # Print the results
    print(f"\nSelected Model: {model_name}")
    print(f"\nGenerated Response:\n{response}")


if __name__ == "__main__":
    main()
