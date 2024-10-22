from transformers import pipeline

# Step 1: Define some example sentences
sentences = [
    "I love studying artificial intelligence!",
    "This class is so boring.",
    "I like the part of the class about machine learning.",
    "I am excited about the new project.",
    "I hate being stuck in traffic.",
    "The food at the cafeteria is average."
]
sarcastic_sentences = [
    "Oh, fantastic! I love getting stuck in traffic.",
    "Wow, you really nailed it... not.",
    "Great job on the project... if failing was the goal."
]

# Step 2: Load a sentiment-analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Step 3: Run sentiment analysis on each sentence
results = sentiment_pipeline(sarcastic_sentences)

# Step 4: Display the results
for sentence, result in zip(sarcastic_sentences, results):
    print(f"'{sentence}' -> Sentiment: {result['label']} (Score: {result['score']:.2f})")
