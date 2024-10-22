import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

# Example sentences
sentences = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "The cat and the dog are friends"
]

# Step 1: Create the vocabulary (set of unique words)
vocabulary = set()
for sentence in sentences:
    vocabulary.update(sentence.lower().split())
vocabulary = sorted(list(vocabulary))

# Step 2: Create the BoW matrix dynamically
bow_matrix = np.zeros((len(sentences), len(vocabulary)))
for i, sentence in enumerate(sentences):
    for word in sentence.lower().split():
        if word in vocabulary:
            j = vocabulary.index(word)
            bow_matrix[i, j] += 1

print("BoW Matrix:")
print(vocabulary)
print(bow_matrix)
print(" ")

# Step 3: Calculate Term Frequency (TF)
tf = bow_matrix / bow_matrix.sum(axis=1, keepdims=True)

# Step 4: Calculate Inverse Document Frequency (IDF) with smoothing
num_docs = len(sentences)
idf = np.log((num_docs) / (1 + (bow_matrix > 0).sum(axis=0)))  # Add 1 for smoothing

# Step 5: Calculate TF-IDF
tf_idf = tf * idf

# Step 6: Create a DataFrame to display the TF-IDF matrix
tf_idf_df = pd.DataFrame(tf_idf, columns=vocabulary, 
                         index=["Sentence 1", "Sentence 2", "Sentence 3"])

# Display the results
print("TF-IDF Matrix:")
print(tf_idf_df)
