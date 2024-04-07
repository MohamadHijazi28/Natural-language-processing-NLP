import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from gensim.models import Word2Vec

random.seed(42)
np.random.seed(42)


def words_only(sentence):
    # Split the sentence into tokens based on space
    tokens = sentence.split()
    # Filter out tokens that are not words
    words = [token for token in tokens if token.isalpha()]
    return words


# Create chunks
def create_chunks(tokenized_sentences, chunk_len):
    chunks = []
    for i in range(0, len(tokenized_sentences), chunk_len):
        chunk = tokenized_sentences[i:i+chunk_len]
        if len(chunk) == chunk_len:
            chunks.append([item for sublist in chunk for item in sublist])  # Flatten list of lists
    return chunks


# Downsampling
def down_sampling(committee_chunks, plenary_chunks):
    if len(committee_chunks) > len(plenary_chunks):
        committee_chunks = random.sample(committee_chunks, len(plenary_chunks))
    else:
        plenary_chunks = random.sample(plenary_chunks, len(committee_chunks))
    return committee_chunks, plenary_chunks


# Generate sentence embeddings
def sentence_embedding(chunk, model):
    valid_tokens = [token for token in chunk if token in model.wv.key_to_index]
    if not valid_tokens:
        return np.zeros(model.vector_size)
    word_vectors = [model.wv[token] for token in valid_tokens]
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector


def train_model(chunk_len, model):
    committee_chunks = create_chunks(committee['tokenized_sentences'].tolist(), chunk_len)
    plenary_chunks = create_chunks(plenary['tokenized_sentences'].tolist(), chunk_len)

    committee_chunks, plenary_chunks = down_sampling(committee_chunks, plenary_chunks)

    # Train Word2Vec model on the downsampled dataset
    all_chunks = committee_chunks + plenary_chunks
    #model = Word2Vec(sentences=all_chunks, vector_size=100, window=5, min_count=1)

    # Apply sentence embedding on chunks
    X = np.array([sentence_embedding(chunk, model) for chunk in all_chunks])
    y = np.array([0] * len(committee_chunks) + [1] * len(plenary_chunks))

    X, y = shuffle(X, y, random_state=42)

    # KNN Classifier
    knn = KNeighborsClassifier()

    # Evaluation with Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("KNN Stratified Train-Test Split Evaluation:\n", classification_report(y_test, y_pred_knn))


# Load CSV
#csv_file_path = 'output_data.csv'
csv_file_path = sys.argv[1]
df = pd.read_csv(csv_file_path, quotechar="$")

df['tokenized_sentences'] = df['sentence_text'].apply(words_only)

# Filter data
committee = df[df['protocol_type'] == 'committee']
plenary = df[df['protocol_type'] == 'plenary']

model_path = sys.argv[2]
model = Word2Vec.load(model_path)

for size in [1, 3, 5]:
    print(f"Chunk size = {size}:")
    train_model(size, model)
