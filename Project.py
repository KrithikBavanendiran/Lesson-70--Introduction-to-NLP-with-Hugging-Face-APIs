from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_message(message):
    """
    Classifies a message as 'Safe' or 'Spam' using zero-shot classification.

    Args:
        message (str): The input message to classify.

    Returns:
        str: The classification result ('Safe' or 'Spam').
    """
    # Use more descriptive candidate labels
    candidate_labels = ["This is a safe message", "This is a spam message"]

    # Perform classification
    result = classifier(message, candidate_labels)

    # Extract the label with the highest score
    classification = result["labels"][0]
    score = result["scores"][0]

    # Map the descriptive labels back to "Safe" or "Spam"
    if classification == "This is a safe message":
        classification = "Safe"
    else:
        classification = "Spam"

    print(f"Message: {message}")
    print(f"Classification: {classification} (Confidence: {score:.2f})")

    return classification

if __name__ == "__main__":
    message = input("Enter a message to classify: ")
    classify_message(message)