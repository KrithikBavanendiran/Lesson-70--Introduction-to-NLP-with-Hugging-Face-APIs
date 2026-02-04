from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

data = [
    "Congratulations! You won a free iPhone. Click the link now!",
    "Hey, are we still meeting for homework later?"
]

labels = ["spam message", "safe message"]

for text in data:
    result = classifier(text, labels)
    label = result["labels"][0]
    confidence = result["scores"][0]

    print(f"Message: {text}")
    print(f"Label: {'Spam' if 'spam' in label else 'Safe'}, Confidence: {confidence:.2f}\n")