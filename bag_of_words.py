# Task Description (LLM Generated):
# Given a list of documents, create a Bag-of-Words (BoW) representation using Python.
# The representation should use term frequency-inverse document frequency (TF-IDF).
# The equation is given where term frequency * inverse docuemnt frequency where
# term frequency = number of terms t in a document / total number of terms in the documents
# inverse document frequency = log(n / df) where n is the number of documents and df is the number of doucments that contain that term

# Input:
# A list of documents, e.g.:
# ['I love Python programming', 
#  'Python is great for data science', 
#  'Python is also a snake']

# Output:
# A NumPy array representing the TF-IDF matrix.
# - Rows correspond to documents (indexed by i).
# - Columns correspond to all unique words in the corpus, sorted alphabetically.
# - Each element is the TF-IDF value of the word in that document.

# Notes:
# - The vocabulary is built from the entire corpus.
# - Words should be normalized (e.g., lowercased).
# - The output must be a NumPy array for consistency and efficiency.

import numpy as np
import math
import re
from collections import defaultdict, Counter

def bag_of_words(documents):
    """
    returns tf-idf weighted bag-of-words matrix for a list of documents.
    """
    n = len(documents)
    word_to_docs = defaultdict(set) # word to number of documents
    word_freq_per_doc = {} # word frequency per documents
    words_set = set() # tracks all words

    # preprocess
    def tokenize(document):
        # Remove punctuation and convert to lowercase
        document = re.sub(r'[^\w\s]', '', document)
        return [word.lower() for word in document.split()] 
    
    for i, doc in enumerate(documents):
        words = tokenize(doc)
        print(f"Document {i} words: {words}")  
        counter = Counter(words)
        word_freq_per_doc[i] = {}
        for word, occurences in counter.items():
            word_to_docs[word].add(i)
            word_freq_per_doc[i][word] = occurences
            words_set.add(word)

    # sorting the words with their index
    print(f"All unique words: {sorted(words_set)}")
    words_list = list(words_set)
    sorted_words = sorted(words_list)
    word_to_index = {word: idx for idx, word in enumerate(sorted_words)}

    def tf_idf(term_frequency, doc_freq):
        return term_frequency * math.log(n / doc_freq, math.e)

    # build TF-IDF matrix
    BoW = np.zeros((n, len(sorted_words)))

    for i in range(n):
        # count all frequency
        total_terms = sum(word_freq_per_doc[i].values())
        for word, count in word_freq_per_doc[i].items(): 
            tf = count / total_terms
            df = len(word_to_docs[word])
            idx = word_to_index[word]
            BoW[i][idx] = tf_idf(tf, df)

    return BoW

if __name__ == '__main__':
    documents = [
        "The cat sat on the mat.",
        "The dog played in the park.",
        "cat and dogs are great pets."
    ]
    answer = bag_of_words(documents)
    print("Shape:", answer.shape)
    print("TF-IDF Matrix:\n", answer)