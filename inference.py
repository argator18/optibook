from transformers import TextClassificationPipeline

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

pipe("I love sunny weather!")  # Output: list of class scores

