import sys

from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def words_only(sentence):
    # Split the sentence into tokens based on space
    tokens = sentence.split()
    # Filter out tokens that are not words;we consider a "word" anything that doesn't contain digits or punctuation
    words = [token for token in tokens if token.isalpha()]
    return words


def sentence_embedding(sentence_tokens, model):
    # Filter out tokens not recognized by the model
    valid_tokens = [token for token in sentence_tokens if token in model.wv.key_to_index]
    # If no valid tokens in sentence, return a zero vector;all tokens that are not valid are empty.
    if not valid_tokens:
        #print("NOT VALID")
        return np.zeros(model.vector_size)
    # Calculate the weighted average of the word vectors
    word_vectors = [model.wv[token] for token in valid_tokens]
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector


# Load the CSV file
#csv_file_path = 'output_data.csv'
csv_file_path = sys.argv[1]
df = pd.read_csv(csv_file_path, quotechar="$")

# Filter non-words tokens
df['filtered_sentences'] = df['sentence_text'].apply(words_only)
tokenized_sentences = df['filtered_sentences'].tolist()

outputs_folder_path = sys.argv[2]

# Create Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1)

# Save the model for future use
model_path = outputs_folder_path + "\\knesset_word2vec.model"
model.save(model_path)

#model = Word2Vec.load(model_path)
word_vectors = model.wv


words_to_compare = ["ישראל", "כנסת", "ממשלה", "חבר", "שלום", "שולחן"]

similar_words = {word: model.wv.most_similar(word, topn=5) for word in words_to_compare}

output_path = outputs_folder_path + "\\knesset_similar_words.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for word, similarities in similar_words.items():
        f.write(f"{word}: {', '.join([f'({sim_word}, {sim_score:.4f})' for sim_word, sim_score in similarities])}\n")


# Apply the function to each row in the DataFrame to create sentence embeddings
df['sentence_embeddings'] = df['filtered_sentences'].apply(lambda tokens: sentence_embedding(tokens, model))

# Store the numpy array of sentence embeddings in a new column 'sentence_embeddings'
sentences_embedding = df['sentence_embeddings'].tolist()

# Ensure that sentence embeddings are in a numpy array
sentence_embeddings = np.array(df['sentence_embeddings'].tolist())

# Select 10 sentences with at least 4 tokens
selected_sentences = df[df['filtered_sentences'].apply(lambda x: 4 <= len(x) < 6)]['sentence_text'].sample(n=10, random_state=7)

# Find the most similar sentence for each selected sentence
most_similar_sentences = {}
for sentence in selected_sentences:
    # Get the index of the selected sentence
    idx = df.index[df['sentence_text'] == sentence].tolist()[0]
    # Compute cosine similarity between the selected sentence and all others
    similarities = cosine_similarity(
        [sentence_embeddings[idx]], # Embedding of the selected sentence
        sentence_embeddings # Embeddings of all sentences
    )[0]
    # Get the index of the most similar sentence (ignoring the selected sentence itself)
    most_similar_idx = similarities.argsort()[-2] # -2 because the most similar is the sentence itself
    # Find the actual sentence text
    most_similar_sentence = df.iloc[most_similar_idx]['sentence_text']
    most_similar_sentences[sentence] = most_similar_sentence

# Write the results to a file
output_file_path = outputs_folder_path + "\\knesset_similar_sentences.txt"
with open(output_file_path, 'w', encoding='utf-8') as file:
    for original, similar in most_similar_sentences.items():
        file.write(f"{original}: most similar sentence: {similar}\n")

sentences = [
    "ברוכים הבאים , הכנסו בבקשה לחדר.",
    "אני מוכנה להאריך את ההסכם באותם תנאים.",
    "בוקר טוב , אני פותח את הישיבה.",
    "שלום , הערב התבשרנו שחברינו היקר לא ימשיך איתנו בשנה הבאה."
]
red_words = [
    ["לחדר"],
    ["מוכנה", "ההסכם"],
    ["טוב", "פותח"],
    ["שלום", "היקר", "בשנה"]
]

new_sentences = sentences.copy()

similar_word = model.wv.most_similar(positive=['כפיים','למחוא'],negative=[], topn=1)
new_sentences[0] = new_sentences[0].replace('לחדר', similar_word[0][0])

similar_word = model.wv.most_similar(positive=['מתכוונת','מוכנה'],negative=[], topn=1)
new_sentences[1] = new_sentences[1].replace('מוכנה', similar_word[0][0])

similar_word = model.wv.most_similar(positive=['ההסכם'],negative=[], topn=3)
new_sentences[1] = new_sentences[1].replace('ההסכם', similar_word[2][0])

similar_word = model.wv.most_similar(positive=['אור','טוב'],negative=[], topn=1)
new_sentences[2] = new_sentences[2].replace('טוב', similar_word[0][0])

similar_word = model.wv.most_similar(positive=['פותח','מתחיל'],negative=[], topn=1)
new_sentences[2] = new_sentences[2].replace('פותח', similar_word[0][0])

similar_word = model.wv.most_similar(positive=['צהריים'],negative=[], topn=1)
new_sentences[3] = new_sentences[3].replace('שלום', similar_word[0][0])

similar_word = model.wv.most_similar(positive=['היקר','הטוב','החבר'],negative=[], topn=1)
new_sentences[3] = new_sentences[3].replace('היקר', similar_word[0][0])

similar_word = model.wv.most_similar(positive=['בשנה'],negative=[], topn=1)
new_sentences[3] = new_sentences[3].replace('בשנה', similar_word[0][0])

# Write the results to a text file
with open(outputs_folder_path + "\\red_words_sentences.txt", "w", encoding="utf-8") as file:
    for i in range(len(new_sentences)):
        # Write the original and new sentence to the file
        file.write(f"{sentences[i]}: {new_sentences[i]}\n")

# Words to compare
word1 = 'לבן'
word2 = 'שחור'

# Compute the distance between the two words
distance = model.wv.distance(word1, word2)
print(f"The distance between '{word1}' and '{word2}' is: {distance}")
