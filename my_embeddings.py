from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example sentences
#sentences = [
#    "The cat sat on the mat.",
#    "The dog sat on the log.",
#    "The cat and the dog are friends."
#]
sentences = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat and the dog are friends.",
    "A bird flew over the log.",
    "The mat is under the table.",
    "The dog chased the cat.",
    "The friends played in the garden.",
    "The cat likes to sleep on the mat."
]

# Step 1: Generate sentence embeddings
sentence_embeddings = model.encode(sentences)

# Step 2: Display the embeddings
for i, embedding in enumerate(sentence_embeddings):
    if i == 0:
        print("Number of dimensions: ", len(embedding))
        print(" ")
    print(f"Sentence {i+1} embedding: {embedding[:10]}...")  # Display first 10 values for brevity

# Step 3: Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(sentence_embeddings)

# Step 4: Plot the reduced embeddings
plt.figure(figsize=(10, 8))
texts = []  # Store text objects for adjustment

for i, sentence in enumerate(sentences):
    x, y = reduced_embeddings[i]
    plt.scatter(x, y, label=f"Sentence {i + 1}", s=50)
    texts.append(plt.text(x, y, sentence, fontsize=9))

# Adjust text labels to avoid overlaps and off-graph labels
adjust_text(texts)

plt.title("2D Visualization of Sentence Embeddings using PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()