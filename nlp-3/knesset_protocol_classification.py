import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle

random.seed(42)
np.random.seed(42)


def create_chunks(sentences, chunk_len):
    chunks = []
    sentences_list = sentences.split('\n')
    for i in range(0, len(sentences), chunk_len):
        chunk = sentences_list[i:i+chunk_len]
        if len(chunk) == chunk_len:
            chunks.append(' '.join(chunk))
    return chunks


# down-sample the larger class by randomly removing chunks
def down_sampling(committee_chunks, plenary_chunks):
    if len(committee_chunks) > len(plenary_chunks):
        committee_chunks = random.sample(committee_chunks, len(plenary_chunks))
    else:
        plenary_chunks = random.sample(plenary_chunks, len(committee_chunks))

    return committee_chunks, plenary_chunks


def extract_features(chunks):
    # Combine all unique keywords from both lists to ensure each is a separate feature
    all_keywords = list(set(plenary_keywords + committee_keywords))

    # Initialize a dictionary to map each keyword to its feature index
    keyword_to_index = {keyword: i for i, keyword in enumerate(all_keywords)}

    # Initialize the features list
    features = []

    for chunk in chunks:
        # Initialize a feature vector for this chunk with zeros for each keyword
        chunk_features = [0] * len(all_keywords)

        # Average sentence length in words
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in chunk if sentence])
        words_count = len(chunk.split())
        # Count occurrences of each keyword in the chunk
        for keyword in all_keywords:
            keyword_count = chunk.count(keyword)
            index = keyword_to_index[keyword]
            chunk_features[index] = keyword_count
        comma_count = sum(sentence.count(',') for sentence in chunk)
        # Combine avg_sentence_length and words_count with keyword features
        full_features = [avg_sentence_length,words_count,comma_count] + chunk_features
        features.append(full_features)

    return np.array(features)


# Keywords list
plenary_keywords = ['במליאה', 'מליאה', 'בחירות', 'מצביעים', 'חברי המליאה', 'הצעת חוק', 'דיון', 'נאום'
    ,'בהתייוונות ובהתבוללות','המורמונים ביוטה','להתערטל מדתי','הטנדנציה הדומיננטית','שההפרדה המלאכותית'
    ,'ויבחרו במצוותיו','שקרב לזבוח','ומודעת ליהדותה','הטועים וסוברים','וכמאמר רבותינו'
                             ,'חבר הכנסת','חברי הכנסת','היושב ראש','אדוני היושב ראש','בסופו של דבר','אדוני היושב','של חבר הכנסת']
committee_keywords = ['בוועדה', 'חברי הוועדה', 'וועדה', 'סדר היום', 'דיון בוועדה', 'תקנות', 'הצעה','לשגר תנחומים'
    ,'אחל לאברום','מצדדים בחיזוק','האבות המייסדים','היעודיים יטפלו','שחוקים שמקנים','שוועדה שעסוקה','קטליזטור וזרז'
    ,'וש"ס כסיעה','כלאם פאדי''תודה','הכנסת'
    , 'את זה', 'אני לא', 'זה לא','אני לא יודע','אני רוצה']

# Load the CSV
csv_file_path = 'output_data.csv'
df = pd.read_csv(csv_file_path, quotechar="$")

# Filter data
committee = df[df['protocol_type'] == 'committee']
committee_corpus = '\n'.join(committee['sentence_text'].astype(str))

plenary = df[df['protocol_type'] == 'plenary']
plenary_corpus = '\n'.join(plenary['sentence_text'].astype(str))

committee_chunks = create_chunks(committee_corpus, 5)
plenary_chunks = create_chunks(plenary_corpus, 5)

committee_chunks, plenary_chunks = down_sampling(committee_chunks, plenary_chunks)

all_chunks = committee_chunks + plenary_chunks
labels = [0] * len(committee_chunks) + [1] * len(plenary_chunks)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_chunks)
y = np.array(labels)

X, y = shuffle(X, y, random_state=42)

# Evaluation with 10-fold Cross-validation and Stratified Train-Test Split
knn = KNeighborsClassifier()
svm = SVC()

# 10-fold CV
cv = StratifiedKFold(n_splits=10)
knn_cv_scores = cross_val_score(knn, X, y, cv=cv, n_jobs=-1)
svm_cv_scores = cross_val_score(svm, X, y, cv=cv, n_jobs=-1)
print("KNN 10-fold CV Accuracy: %0.2f" % (knn_cv_scores.mean()))
print("SVM 10-fold CV Accuracy: %0.2f" % (svm_cv_scores.mean()))

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# KNN
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Stratified Train-Test Split Evaluation:\n", classification_report(y_test, y_pred_knn))

# SVM
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Stratified Train-Test Split Evaluation:\n", classification_report(y_test, y_pred_svm))

#################################################################################################
# Use the existing all_chunks for feature extraction
all_chunks_features = extract_features(all_chunks)

# Shuffle the feature set
X_new_features, y_new = shuffle(all_chunks_features, np.array(labels), random_state=42)

# Stratified Train-Test Split
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new_features, y_new, test_size=0.1, stratify=y_new, random_state=42)

# Initialize models
knn_new = KNeighborsClassifier()
svm_new = SVC()

# Evaluation with 10-fold CV
cv_new = StratifiedKFold(n_splits=10)
knn_cv_scores_new = cross_val_score(knn_new, X_train_new, y_train_new, cv=cv_new, n_jobs=-1)
svm_cv_scores_new = cross_val_score(svm_new, X_train_new, y_train_new, cv=cv_new, n_jobs=-1)
print("KNN 10-fold CV Accuracy with Our Features: %0.2f" % (knn_cv_scores_new.mean()))
print("SVM 10-fold CV Accuracy with Our Features: %0.2f" % (svm_cv_scores_new.mean()))

# Train and evaluate with Stratified Train-Test Split using our features
# KNN
knn_new.fit(X_train_new, y_train_new)
y_pred_knn_new = knn_new.predict(X_test_new)
print("KNN Stratified Train-Test Split Evaluation with Our Features:\n", classification_report(y_test_new, y_pred_knn_new))

# SVM
svm_new.fit(X_train_new, y_train_new)
y_pred_svm_new = svm_new.predict(X_test_new)
print("SVM Stratified Train-Test Split Evaluation with Our Features:\n", classification_report(y_test_new, y_pred_svm_new))

# Load knesset_text_chunks.txt
with open('knesset_text_chunks.txt', 'r', encoding='utf-8') as file:
    new_chunks = file.readlines()
new_chunks_transformed = vectorizer.transform(new_chunks)

# Using SVM (the best-performing) model for prediction
predictions = svm.predict(new_chunks_transformed)
class_labels = ['committee' if pred == 0 else 'plenary' for pred in predictions]

# Write results
with open('classification_results_test.txt', 'w', encoding='utf-8') as file:
    for label in class_labels:
        file.write(label + '\n')
