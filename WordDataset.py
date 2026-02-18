from torch.utils.data import IterableDataset
from vocabulary import Vocabulary
import auxfunctions as aux
import torch

class TextLSTM:
    @staticmethod
    def _pad_sequences(sequences,max_length,pad_value):
        """
        Docstring for _pad_sequences

        :param sequences: Input list of sequences to be padded
        :param max_length: Desired length of the output sequences after padding
        :param pad_value: Value to be used for padding the sequences, which is typically a special token index representing padding in the vocabulary.
        :return: List of padded sequences, where each sequence is of length max_length and shorter sequences are padded with the specified pad_value.
        """
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_length:
                padded_seq = seq + [pad_value] * (max_length - len(seq))
            else:
                padded_seq = seq[:max_length]
            padded_sequences.append(padded_seq)
        return padded_sequences
    
    @staticmethod
    def split_article_into_sequences(article, vocabulary, min_sequence_length, max_sequence_length, step=1):
        """
        Docstring for split_article_into_sequences
    
        :param article: Input article to be split into sequences
        :param vocabulary: Vocabulary object to be used for preprocessing the article
        :param min_sequence_length: Minimum length of the sequences to be generated from the article
        :param max_sequence_length: Maximum length of the sequences to be generated from the article
        :param step: Step size for generating sequences, which determines how much the window moves when creating new sequences from the article. A step of 1 means that the window will move one word at a time, while a larger step will create fewer sequences by skipping some words in between.
        :return:
         current_word_targets: List of sequences of word indices representing the current words in the article, where each sequence is of length between min_sequence_length and max_sequence_length.
         next_word_targets: List of word indices representing the next word in the article corresponding to each sequence in current_word_targets, where each index corresponds to the word that follows the last word in the
        """
        article=aux.preprocess_article(article, vocabulary)
        current_word_targets=[]
        next_word_targets=[]
        
        #Sliding window approach


        for i in range(0, len(article) - min_sequence_length,step):
            for j in range(i + min_sequence_length, min(i + max_sequence_length, len(article)-1) + 1, step):
                current_word_targets.append(article[i:j])
                next_word_targets.append(article[j+1])

        return current_word_targets, next_word_targets


#We choose IterableDataset because it allows us to create a dataset that can be iterated over in a streaming fashion, which is particularly useful when dealing with large datasets that may not fit entirely into memory. By using an IterableDataset, we can generate and process data on-the-fly during training, which can help to reduce memory usage and improve efficiency when working with large text corpora.
class WordDataset(IterableDataset):
    def __init__(self,articles_generator, vocabulary, minimum_sequence_length, maximum_sequence_length, step=1):
        """
        Docstring for __init__
        
        :param articles_generator: Generator that yields articles (strings)
        :param vocabulary: Vocabulary object to be used for preprocessing the articles
        :param minimum_sequence_length: Minimum length of the sequences to be generated from each article
        :param maximum_sequence_length: Maximum length of the sequences to be generated from each article
        :param step: Step size for generating sequences, which determines how much the window moves when creating new sequences from an article. A step of 1 means that the window will move one word at a time, while a larger step will create fewer sequences by skipping some words in between.
        """
        self.articles_generator=articles_generator
        self.vocabulary=vocabulary
        self.minimum_sequence_length=minimum_sequence_length
        self.maximum_sequence_length=maximum_sequence_length
        self.step=step
    def __iter__(self):
        for article in self.articles_generator:
            current_word_targets, next_word_targets = TextLSTM.split_article_into_sequences(
                article,
                self.vocabulary,
                self.minimum_sequence_length,
                self.maximum_sequence_length,
                self.step
            )
            for current, next in zip(current_word_targets, next_word_targets):
                yield current, next
    
def collate_fn(vocabulary,batch):
        """
        Docstring for collate_fn
        
        :param batch: List of tuples, where each tuple contains a sequence of word indices representing the current words in an article and a corresponding word index representing the next word in the article.
        :return: Tuple containing three tensors:
            - A tensor of shape (batch_size, max_sequence_length) containing the padded sequences of word indices representing the current words in the articles, where max_sequence_length is the length of the longest sequence in the batch after padding.
            - A tensor of shape (batch_size,) containing the word indices representing the next words in the articles corresponding to each sequence in the batch.
            - A tensor of shape (batch_size,) containing the original lengths of the sequences before padding, which can be used for masking or other purposes during model training.
        """
        current_word_targets, next_word_targets = zip(*batch)
        max_length = max(len(seq) for seq in current_word_targets)
        lengths = [len(seq) for seq in current_word_targets]
        padded_current_word_targets = TextLSTM._pad_sequences(current_word_targets, max_length, vocabulary.pad_index)
        return torch.tensor(padded_current_word_targets), torch.tensor(next_word_targets),torch.tensor(lengths)




