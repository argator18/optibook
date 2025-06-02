from transformers import pipeline

# Load pre-trained zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Example usage
sequence = "The stock market crashed today."
labels = ["Finance", "Sports", "Politics", "Technology"]

result = classifier(sequence, labels)
print(result)

