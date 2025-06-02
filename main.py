import pandas as pd
from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load the data
df = pd.read_csv("trainings.csv")  # Replace with your actual filename

# Extract column names
stock_columns = ['NVDA', 'ING', 'SAN', 'PFE', 'CSCO']
text_column = 'SocialMediaFeed'

# Track statistics
correct = 0
wrong = 0
skipped = 0

for idx, row in df.iterrows():
    print(idx,end="\r")
    if idx > 200:
        break
    text = row[text_column]
    true_affected_stocks = [stock for stock in stock_columns if row[stock] != 0.0]

    # Skip posts with no affected stocks
    if not true_affected_stocks:
        skipped += 1
        continue

    # Predict the most likely stock affected
    result = classifier(text, stock_columns)
    predicted_stock = result['labels'][0]

    if predicted_stock in true_affected_stocks:
        correct += 1
    else:
        wrong += 1

# Print results
total = correct + wrong
print(f"Total classified: {total}")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {wrong}")
print(f"Accuracy: {correct / total:.2%}")
print(f"Skipped neutral posts: {skipped}")
