import pandas as pd
import re
from collections import Counter

def clean_texts(texts):
    """
    Docstring for clean_texts
    
    :param texts: Input text batch to be cleaned
    :return: Cleaned text batch
    """
    # Convert to lowercase
    texts = texts.str.lower()
    # Remove punctuation and special characters using regex
    texts = texts.str.replace(r'[^\w\s]', '', regex=True) # Remove punctuation and special characters
    texts = texts.str.strip() # Remove leading and trailing whitespace
    texts = texts.replace('', '<UNK>') # Replace empty strings with a special token
    return texts
"""
We attempt to vectorize the text data in order to make it more efficient, however as the dataset is quite large
we will not be able to vectorize the entire dataset, instead we will vectorize by batches 
in order to avoid memory issues.
"""
def build_vocabulary(dataframe, text_column='text'):
    """
    Docstring for build_vocabulary
    
    :param dataframe: DataFrame containing the text data
    :param text_column: Name of the column containing the text
    :return: vocabularyToIndex, indexToVocabulary
    """
    additional=["UNK", "PAD"] # Add special tokens for padding and unknown words
    word_counter = Counter()
    for start_idx in range(0, len(dataframe), 1000): # Process the data in batches of 1,000 rows
        end_idx = min(start_idx + 1000, len(dataframe))
        batch = dataframe.iloc[start_idx:end_idx]
        clean_batch = clean_texts(batch[text_column]) # Clean the text in the batch
        all_words = clean_batch.str.split() # Split the cleaned text into individual words
        for words in all_words:
            word_counter.update(words) # Update the word frequency counter with the words from the batch
    # Create a vocabulary dictionary mapping each word to a unique index, starting from 2 to reserve 0 and 1 for special tokens
    vocabularyToIndex = {word: idx+2 for idx, (word, count) in enumerate(word_counter.most_common())} 
    for i, token in enumerate(additional):
        vocabularyToIndex[token] = i # Add special tokens to the vocabulary with reserved indices
    # Create a reverse mapping from index to word (optional, but useful for decoding)
    indexToVocabulary = {idx: word for word, idx in vocabularyToIndex.items()} # Create a reverse mapping from index to word
    vocabulary_size = len(vocabularyToIndex) # Calculate the size of the vocabulary
    print(f"Vocabulary size: {vocabulary_size}")
    return vocabularyToIndex, indexToVocabulary # Return the vocabulary mappings

def text_to_sequence(text, vocabularyToIndex):
    """
    Docstring for text_to_sequence
    
    :param text: Input text to be converted to a sequence of indices
    :param vocabularyToIndex: Dictionary mapping words to their corresponding indices
    :return: List of indices representing the input text
    """
    clean_text = clean_texts(pd.Series([text]))[0] # Clean the input text
    words = clean_text.split() # Split the cleaned text into individual words
    sequence = [vocabularyToIndex.get(word, vocabularyToIndex["UNK"]) for word in words] # Convert words to indices, using UNK for unknown words
    return sequence # Return the list of indices representing the input text

vocabulary1, indexToVocabulary1 = build_vocabulary(pd.read_parquet("data/train.parquet"))
print(f"Sample vocabulary entries: {list(vocabulary1.items())[:10]}")
print(len(pd.read_parquet("data/train.parquet")))
print(text_to_sequence("This is a sample text to be converted to a sequence of indices.", vocabulary1))