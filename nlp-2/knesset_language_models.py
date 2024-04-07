import math
import sys

import pandas as pd
from collections import defaultdict, Counter, OrderedDict
import time


class Trigram_LM:
    def __init__(self, corpus):
        self.corpus = corpus
        self.unigram_model = self.build_unigram_model()
        self.bigram_model = self.build_bigram_model()
        self.trigram_model = self.build_trigram_model()

    def preprocess_corpus(self):
        # Tokenize the corpus into words
        tokenized_corpus = self.corpus.split(' ')
        return tokenized_corpus

    def build_unigram_model(self):
        model = Counter()
        tokenized_corpus = self.preprocess_corpus()
        for token in tokenized_corpus:
            model[token] += 1
        return model

    def build_bigram_model(self):
        model = defaultdict(Counter)
        tokenized_corpus = self.preprocess_corpus()
        for i in range(len(tokenized_corpus) - 1):
            current_token = tokenized_corpus[i]
            next_token = tokenized_corpus[i + 1]
            model[current_token][next_token] += 1
        return model

    def build_trigram_model(self):
        model = defaultdict(Counter)
        tokenized_corpus = self.preprocess_corpus()
        for i in range(len(tokenized_corpus) - 2):
            current_token = tokenized_corpus[i]
            next_token = tokenized_corpus[i + 1]
            next_next_token = tokenized_corpus[i + 2]
            model[current_token, next_token][next_next_token] += 1
        return model

    def calculate_prob_of_sentence(self, sentence, smoothing):
        prob = 1.000
        lambda1, lambda2, lambda3 = 0.1, 0.3, 0.6
        sentence = sentence.split()
        n = len(sentence)
        vocabulary_size = len(self.unigram_model)

        if n == 1:
            word = sentence[0]
            unigram_prob = (self.unigram_model.get(word, 0) + 1) / (sum(self.unigram_model.values()) + vocabulary_size)
            prob *= unigram_prob
        elif n == 2:
            context = sentence[0]
            word = sentence[1]
            bigram_context_count = sum(self.bigram_model[context].values())
            bigram_prob = (self.bigram_model[context].get(word, 0) + 1) / (bigram_context_count + vocabulary_size)
            prob *= bigram_prob
        else:
            for i in range(n - 2):
                context = (sentence[i], sentence[i + 1])
                word = sentence[i + 2]
                trigram_context_count = sum(self.trigram_model[context].values())
                word_count = self.trigram_model[context].get(word, 0)
                if smoothing == 'Linear':
                    unigram_prob = (self.unigram_model.get(word, 0) + 1) / (
                            sum(self.unigram_model.values()) + vocabulary_size)
                    # Correctly use the first word of the bigram as context for bigram_prob
                    bigram_context = sentence[i]
                    bigram_prob = (self.bigram_model[bigram_context].get(sentence[i + 1], 0) + 1) / (
                            sum(self.bigram_model[bigram_context].values()) + vocabulary_size)
                    trigram_prob = (word_count + 1) / (trigram_context_count + vocabulary_size)
                    prob *= (lambda1 * unigram_prob) + (lambda2 * bigram_prob) + (lambda3 * trigram_prob)
                elif smoothing == 'Laplace':
                    prob *= (word_count + 1) / (trigram_context_count + vocabulary_size)
        log_prob = math.log(prob) if prob > 0 else float('-inf')  # Check if prob is greater than 0 to avoid error

        print(f'The log probability is: {round(log_prob, 3)}')
        return round(log_prob, 3)

    def generate_next_token(self, sentence):
        if isinstance(sentence, list):
            tokenized_context = sentence
        else:
            tokenized_context = sentence.split(" ")
        if len(tokenized_context) >= 2:  # use trigram model
            w1, w2 = tokenized_context[-2:]
            next_tokens_counts = self.trigram_model[w1, w2]
        elif len(tokenized_context) == 1:  # use bigram model
            w1 = tokenized_context[-1]
            next_tokens_counts = self.bigram_model[w1]
        else:  # Use unigram model
            next_tokens_counts = self.unigram_model['<s>']

        # Find the token with the maximum count
        if next_tokens_counts:
            next_token = self.get_best_token_from_list_tokens(sentence, tokenized_context, next_tokens_counts.keys())
            print(f'The next token is: {next_token}')
            return next_token
        else:
            possible_tokens = self.unigram_model
            ordered_tokens = OrderedDict(sorted(possible_tokens.items(), key=lambda item: item[1], reverse=True))
            tokens = [token for token in ordered_tokens if token.isalpha()]
            best_token = self.get_best_token_from_list_tokens(sentence, tokenized_context, tokens)
            print(f'The next token is: {best_token}')
            return best_token

    def get_best_token_from_list_tokens(self, sentence, tokenized_context, list_tokens):
        best_token = None
        best_probability = float('-inf')
        for token in list_tokens:
            if isinstance(sentence, list):
                sentence_with_token = tokenized_context + [token]
                sentence_with_token = " ".join(sentence_with_token)
            else:
                sentence_with_token = sentence + " " + token
            probability = self.calculate_prob_of_sentence(sentence_with_token, 'Linear')
            if probability > best_probability:
                best_token = token
                best_probability = probability

        return best_token

    def split_into_sentences(self):
        sentence_endings = ['.', '!', '?']
        sentences = []
        start_index = 0
        for i, char in enumerate(self.corpus):
            if char in sentence_endings:
                # Check if the character is a sentence-ending punctuation mark
                sentences.append(self.corpus[start_index:i + 1].strip())
                start_index = i + 1

        if start_index < len(self.corpus):
            sentences.append(self.corpus[start_index:].strip())

        return sentences

    def get_k_n_collocations(self, k, n):
        # Initialize frequency dictionaries and total counts
        unigram_freqs = defaultdict(int)
        ngram_freqs = defaultdict(int)
        total_words = 0

        # Count unigrams, n-grams, and total words
        sentences = self.split_into_sentences()
        for sentence in sentences:
            sentence = sentence.split()
            total_words += len(sentence)
            for word in sentence:
                unigram_freqs[word] += 1
            for i in range(len(sentence) - n + 1):
                ngram = tuple(sentence[i:i + n])
                ngram_freqs[ngram] += 1

        # Calculate PMI for each n-gram
        pmi_scores = []
        for ngram, ngram_freq in ngram_freqs.items():
            p_ngram = ngram_freq / total_words
            p_individual = 1
            for word in ngram:
                p_individual *= unigram_freqs[word] / total_words
            pmi = math.log(p_ngram / p_individual, 2)
            pmi_scores.append((ngram, pmi))

        # Sort the list by PMI score in descending order
        pmi_scores.sort(key=lambda x: x[1], reverse=True)
        return pmi_scores[:k]

    def find_token(self, sentence):
        token = None
        tokens_list = []
        original_sentence = sentence.split()
        stars_index = [i for i, x in enumerate(original_sentence) if x == '[*]']
        for star_index in stars_index:
            if star_index < 1:
                # use a unigram model if the '[*]' is the first word
                possible_words = list(self.unigram_model.keys())
                word_probs = [self.unigram_model[word] / sum(self.unigram_model.values()) for word in possible_words]
                if not word_probs:
                    most_probable_word = ''
                else:
                    most_probable_word = possible_words[word_probs.index(max(word_probs))]
                    token = most_probable_word
                original_sentence[star_index] = most_probable_word
            elif star_index == 1:
                # use a bigram model if the '[*]' is the second word
                context = original_sentence[star_index - 1]
                token = self.generate_next_token(context)

            else:
                # use a trigram model if the '[*]' is not the first or second word
                context = original_sentence[star_index - 2:star_index]
                token = self.generate_next_token(context)

            tokens_list.append(token)
            original_sentence[star_index] = token

        return ' '.join(original_sentence), tokens_list


def create_knesset_collocations_file(committee_model, plenary_model, output_file, k=10):
    with open(output_file, 'w', encoding='utf-8') as file:
        for n in range(2, 5):  # For 2, 3, and 4-gram collocations
            committee_collocations = committee_model.get_k_n_collocations(k, n)
            plenary_collocations = plenary_model.get_k_n_collocations(k, n)
            if n == 2:
                file.write('Two-gram collocations: \n')
            elif n == 3:
                file.write('Three-gram collocations: \n')
            elif n == 4:
                file.write('Four-gram collocations: \n')
            file.write('Committee corpus: \n')
            for collocation, pmi in committee_collocations:
                file.write(' '.join(collocation) + '\n')
            file.write('\nPlenary corpus: \n')
            for collocation, pmi in plenary_collocations:
                file.write(' '.join(collocation) + '\n')
            file.write(f'\n')


def split_file_to_sentences(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            sentences = file.readlines()
    except UnicodeDecodeError:
        print(f'Error reading file {file_name}.')
        sentences = []

    sentences = [sentence.strip() for sentence in sentences]
    return sentences


def create_sentences_results_file(committee_model, plenary_model, file_name, output_file_name):
    file_sentences = split_file_to_sentences(file_name)
    with open(output_file_name, 'w', encoding='utf-8') as file:
        for sentence in file_sentences:
            file.write(f'Original sentence: {sentence} \n')
            modified_sentence, tokens_list = committee_model.find_token(sentence)
            file.write(f'Committee sentence: {modified_sentence} \n')
            file.write(f'Committee tokens: {", ".join(tokens_list)} \n')
            committee_prob = committee_model.calculate_prob_of_sentence(modified_sentence, "Linear")
            plenary_prob = plenary_model.calculate_prob_of_sentence(modified_sentence, "Linear")
            file.write(f'Probability of committee sentence in committee corpus: {committee_prob} \n')
            file.write(f'Probability of committee sentence in plenary corpus: {plenary_prob} \n')
            if committee_prob > plenary_prob:
                file.write(f'This sentence is more likely to appear in corpus: committee \n')
            else:
                file.write(f'This sentence is more likely to appear in corpus: plenary \n')

            modified_sentence, tokens_list = plenary_model.find_token(sentence)
            file.write(f'Plenary sentence: {modified_sentence} \n')
            file.write(f'Plenary tokens: {", ".join(tokens_list)} \n')
            plenary_prob = plenary_model.calculate_prob_of_sentence(modified_sentence, "Linear")
            committee_prob = committee_model.calculate_prob_of_sentence(modified_sentence, "Linear")
            file.write(f'Probability of plenary sentence in plenary corpus: {plenary_prob} \n')
            file.write(f'Probability of plenary sentence in committee corpus: {committee_prob} \n')
            if committee_prob > plenary_prob:
                file.write(f'This sentence is more likely to appear in corpus: committee \n')
            else:
                file.write(f'This sentence is more likely to appear in corpus: plenary \n')

            file.write(f'\n')


# Loading + processing data and building models
#csv_file_path = 'output_data.csv'
csv_file_path = sys.argv[1]

df = pd.read_csv(csv_file_path, quotechar="$")
committee = df[df['protocol_type'] == 'committee']
committee_corpus = ' '.join(committee['sentence_text'].astype(str))
committee_model = Trigram_LM(committee_corpus)

plenary = df[df['protocol_type'] == 'plenary']
plenary_corpus = ' '.join(plenary['sentence_text'].astype(str))
plenary_model = Trigram_LM(plenary_corpus)

start_time = time.time()

# שלב 2
collocations_knesset_output_file = 'knesset_collocations.txt'
create_knesset_collocations_file(committee_model, plenary_model, collocations_knesset_output_file)
end_time = time.time()
print('Total time for step 2: ', round(end_time - start_time, 3), 'seconds')

# שלב 3
#file_name = 'masked_sentences.txt'
file_name = sys.argv[2]
output_file_name = 'sentences_results.txt'
create_sentences_results_file(committee_model, plenary_model, file_name, output_file_name)
end_time = time.time()

print('Total time for step 3: ', round((end_time - start_time)/60, 3), 'minutes')