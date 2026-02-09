import pandas as pd
import re
from collections import Counter



"""
We attempt to vectorize the text data in order to make it more efficient, however as the dataset is quite large
we will not be able to vectorize the entire dataset, instead we will vectorize by batches 
in order to avoid memory issues.
"""
    

class Vocabulary:
    def __init__(self, dataframe=None, text_column='text'):
        if dataframe is None:
            dataframe = pd.read_parquet("data/train.parquet")
        self._vocabularyToIndex, self._indexToVocabulary,self._length = self._build_vocabulary(dataframe, text_column)
        self._unk_index = self._vocabularyToIndex["UNK"]
        self._pad_index = self._vocabularyToIndex["PAD"]

    @property 
    def unk_index(self):
        return self._unk_index
    
    @property
    def pad_index(self):
        return self._pad_index
    
    @property
    def vocabularyToIndex(self):
        return self._vocabularyToIndex
    

    @property
    def indexToVocabulary(self):
        return self._indexToVocabulary
    
    def __len__(self):
        return self._length
    
    @staticmethod 
    def _clean_texts(texts):
        """
        Docstring for clean_texts
    
        :param texts: Input text batch to be cleaned
        :return: Cleaned text batch type: pandas Series
        """
        # Convert to lowercase
        texts = texts.str.lower()

        # Remove punctuation and special characters using regex
        texts = texts.str.replace(r'[^\w\s]', '', regex=True)

        # Remove leading and trailing whitespace
        texts = texts.str.strip() 

        # Replace empty strings with a special token
        texts = texts.replace('', '<UNK>') 

        return texts
    
    def pad_sequence(self, sequence, max_length):
        """
        Docstring for pad_sequence
    
        :param sequence: Input sequence of word indices to be padded
        :param max_length: Desired length of the output sequence after padding
        :return: Padded sequence of word indices
        """
        # Pad the sequence with the index of the "PAD" token until it reaches the desired max_length
        padded_sequence = sequence + [self._vocabularyToIndex["PAD"]] * (max_length - len(sequence)) 

        # Truncate the sequence if it exceeds the max_length
        return padded_sequence[:max_length]
    
    def text_to_sequence(self, text):
        """
        Docstring for text_to_sequence
    
        :param text: Input text to be converted to a sequence of indices
        :param vocabularyToIndex: Dictionary mapping words to their corresponding indices
        :return: List of indices representing the input text
        """

        # Clean the input text
        clean_text = Vocabulary._clean_texts(pd.Series([text]))[0] 

        # Split the cleaned text into individual words
        words = clean_text.split() 

        # Convert words to indices, using UNK for unknown words
        sequence = [self._vocabularyToIndex.get(word, self._vocabularyToIndex["UNK"]) for word in words] 

        # Return the list of indices representing the input text
        return sequence         
    

    def sequence_to_text(self, sequence):
        return ' '.join([self._indexToVocabulary.get(idx, "UNK") for idx in sequence])
    
    @classmethod
    def _build_vocabulary(cls,dataframe, text_column='text'):
        """
        Docstring for build_vocabulary
    
        :param dataframe: DataFrame containing the text data
        :param text_column: Name of the column containing the text
        :return: vocabularyToIndex, indexToVocabulary, vocabulary_size - dictionaries mapping words to indices and vice versa, and the size of the vocabulary
        """
        # Add special tokens for padding and unknown words
        additional=["UNK", "PAD"] 
        word_counter = Counter()

        # Process the data in batches of 1,000 rows to manage memory usage
        for start_idx in range(0, len(dataframe), 1000): 
            end_idx = min(start_idx + 1000, len(dataframe))
            batch = dataframe.iloc[start_idx:end_idx]

            # Clean the text in the batch
            clean_batch = cls._clean_texts(batch[text_column]) 

            # Split the cleaned text into individual words
            all_words = clean_batch.str.split() 
            for words in all_words:
                word_counter.update(words) # Update the word frequency counter with the words from the batch
    
        # Create a vocabulary dictionary mapping each word to a unique index, starting from 2 to reserve 0 and 1 for special tokens
        vocabularyToIndex = {word: idx+2 for idx, (word, count) in enumerate(word_counter.most_common())} 

        # Add special tokens to the vocabulary with reserved indices
        for i, token in enumerate(additional):
            vocabularyToIndex[token] = i 

        # Create a reverse mapping from index to word (optional, but useful for decoding)
        indexToVocabulary = {idx: word for word, idx in vocabularyToIndex.items()}

        # Calculate the size of the vocabulary
        vocabulary_size = len(vocabularyToIndex) 

        # Return the vocabulary mappings
        return vocabularyToIndex, indexToVocabulary,vocabulary_size
    