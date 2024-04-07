import math
import sys

import pandas as pd
import matplotlib.pyplot as plt
import string

csv_file_path = input("Enter csv file path: ")
#csv_file_path = sys.argv[1]
output_path = input("Enter output path: ")
#output_path = sys.argv[2]
df = pd.read_csv(csv_file_path, quotechar="$")
corpus = ' '.join(df['sentence_text'].astype(str))

# Tokenization
translator = str.maketrans('', '', string.punctuation)
tokens = [word.translate(translator) for word in corpus.split() if word.isalpha()]

# Calculate token frequencies
token_counts = pd.Series(tokens).value_counts()

# Calculate rank and frequency for Zipf's law
rank = list(range(1, len(token_counts) + 1))
frequency = token_counts.to_list()

# Plotting results
plt.plot([math.log(r) for r in rank], [math.log(f) for f in frequency], linestyle='-', color='b')
plt.title('Zipf\'s Law')
plt.xlabel('log(rank)')
plt.ylabel('log(frequency)')
plt.savefig(output_path + '\\output_plot.png')
plt.show()

print("Top 10 words with highest frequency:")
print(token_counts.head(10).reset_index().to_string(header=False, index=False))
print("\nBottom 10 words with lowest frequency:")
print(token_counts.tail(10).reset_index().to_string(header=False, index=False))
