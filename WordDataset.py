from torch.utils.data import IterableDataset
from vocabulary import Vocabulary
import auxfunctions as aux
import torch

class TextTransformer:
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
    


#We choose IterableDataset because it allows us to create a dataset that can be iterated over in a streaming fashion, which is particularly useful when dealing with large datasets that may not fit entirely into memory. By using an IterableDataset, we can generate and process data on-the-fly during training, which can help to reduce memory usage and improve efficiency when working with large text corpora.
class WordDataset(IterableDataset):
    def __init__(self,articles_generator, vocabulary,sequence_length=256, chunksize=1000):
        """
        Docstring for __init__
        
        :param articles_generator: Generator that yields articles (strings)
        :param vocabulary: Vocabulary object to be used for preprocessing the articles
        :param sequence_length: Length of the sequences to be generated from each article
        :param step: Step size for generating sequences, which determines how much the window moves when creating new sequences from an article. A step of 1 means that the window will move one word at a time, while a larger step will create fewer sequences by skipping some words in between.
        """
        self.articles_generator=articles_generator
        self.vocabulary=vocabulary
        self.sequence_length=sequence_length
        self.chunksize=chunksize


    def __iter__(self):
        """
       yield a tuple containing the current word targets, next word targets, attention mask in the dataset. The function processes the articles in a streaming fashion, allowing for memory-efficient handling of large datasets. It uses the vocabulary to preprocess the articles and generates sequences of word indices for training a language model.
        """
        worker_info=torch.utils.data.get_worker_info()
        if worker_info is None:
            iterator=self.articles_generator
        else:
            worker_id=worker_info.id
            num_workers=worker_info.num_workers
            iterator=aux.split_generator(self.articles_generator,num_workers,worker_id)
        for article in iterator:
            index=aux.preprocess_article(article,self.vocabulary)
            #Stride of sequence_length//2 to generate overlapping sequences in order to learn better dependencies between words
            for i in range(0,len(index)-1,self.sequence_length//2):
                current_end=min(i+self.sequence_length,len(index)-1)
                next_end=min(i+self.sequence_length+1,len(index))
                #We skip sequences that are too short to ensure that the model learns from sufficiently long contexts
                if(current_end-i<self.sequence_length//4):
                    continue
                current_word_targets=index[i:current_end]
                next_word_targets=index[i+1:next_end]
                
                yield current_word_targets, next_word_targets
                
    
def collate_fn(vocabulary,batch):
        """
        Docstring for collate_fn
        
        :param batch: List of tuples, where each tuple contains a sequence of word indices representing the current words in an article and a corresponding word index representing the next word in the article.
        :return: Tuple containing four tensors:
            - A tensor of shape (batch_size, max_sequence_length) containing the padded sequences of word indices representing the current words in the articles, where max_sequence_length is the length of the longest sequence in the batch after padding.
            - A tensor of shape (batch_size, max_sequence_length) containing the padded sequences of word indices representing the next words in the articles corresponding to each sequence in the batch.
            - A tensor of shape (batch_size, max_sequence_length) containing the attention masks for each sequence in the batch, where 1 indicates that the position should be attended to and 0 indicates that it should be ignored (e.g., for padding positions).
            - A tensor of shape (batch_size,) containing the original lengths of the current sequences before padding, which can be used for masking or other purposes during model training.
            - A tensor of shape (batch_size,) containing the original lengths of the next word sequences before padding, which can be used for masking or other purposes during model training.
        """
        current_word_targets, next_word_targets = zip(*batch)
        max_length = max(len(seq) for seq in current_word_targets)
        lengths_x = [len(seq) for seq in current_word_targets]
        lengths_y = [len(seq) for seq in next_word_targets]
        attention_masks = []
        for i in range(len(current_word_targets)):
            attention_masks.append([1] * lengths_x[i] + [0] * (max_length - lengths_x[i]))
        padded_current_word_targets = TextTransformer._pad_sequences(current_word_targets, max_length, vocabulary.pad_index)
        padded_next_word_targets = TextTransformer._pad_sequences(next_word_targets, max_length, vocabulary.pad_index)

        return torch.tensor(padded_current_word_targets), torch.tensor(padded_next_word_targets), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(lengths_x), torch.tensor(lengths_y)




