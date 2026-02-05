import pandas as pd
import re
from collections import Counter

def clean_text(text):
    """
    Docstring for clean_text
    
    :param text: Input text to be cleaned
    :return: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters using regex
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation and special characters
    text=text.strip() # Remove leading and trailing whitespace
    if not text: # Check if the cleaned text is empty
        return "<UNK>" # Return a special token for unknown words if the cleaned text is empty
    return text 

def build_vocabulary(dataframe, text_column='text'):
    """
    Docstring for build_vocabulary
    
    :param dataframe: DataFrame containing the text data
    :param text_column: Name of the column containing the text
    :return: vocabularyToIndex, indexToVocabulary
    """
    additional=["UNK", "PAD"] # Add special tokens for padding and unknown words
    word_counter = Counter()
    for index, row in dataframe.iterrows(): #iterate over the rows of the dataframe
        text=row[text_column]
        words = text.split() #split the text into words
        for i in range(len(words)):
            words[i] = clean_text(words[i]) #clean each word
        word_counter.update(words) #update the counter with the words
    # Create a vocabulary dictionary mapping each word to a unique index
    vocabularyToIndex = {word: idx+2 for idx, (word, count) in enumerate(word_counter.most_common())} # Start indexing from 2 to reserve 0 and 1 for special tokens
    # Add special tokens to the vocabulary
    for i, token in enumerate(additional):
        vocabularyToIndex[token] = i # Add special tokens to the vocabulary with reserved indices
    # Create a reverse mapping from index to word (optional, but useful for decoding)
    indexToVocabulary = {idx: word for word, idx in vocabularyToIndex.items()} # Create a reverse mapping from index to word
    vocabulary_size = len(vocabularyToIndex) # Calculate the size of the vocabulary
    print(f"Vocabulary size: {vocabulary_size}")
    return vocabularyToIndex, indexToVocabulary # Return the vocabulary mappings


vocabulary1, indexToVocabulary1 = build_vocabulary(pd.read_parquet("data/train.parquet"))
print(f"Sample vocabulary entries: {list(vocabulary1.items())[:10]}")
    