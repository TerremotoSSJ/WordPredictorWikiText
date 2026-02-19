
from vocabulary import Vocabulary
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class NextWordPredictor(nn.Module):
    """
    LSTM-based model for next word prediction in a sequence of text.
    """

    def __init__(self, vocabulary, embedding_dim, hidden_dim, num_layers=4,dropout=0.2):
        super(NextWordPredictor, self).__init__()

        #Embedding layer converts word indices into dense vectors of fixed size (embedding_dim).
        #padding_idx is set to the index of the padding token in the vocabulary, which allows the model to ignore the padded values during training and inference.
        self.embedding = nn.Embedding(len(vocabulary), embedding_dim, padding_idx=vocabulary.pad_index)

        #Dropout is added to prevent overfitting by randomly setting a fraction of the input units to 0 during training.
        #Batch_first=True ensures that the input and output tensors are of shape (batch_size, sequence_length, hidden_dim).
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,num_layers,batch_first=True,dropout=dropout if dropout > 0 else 0) 
        self.fc=nn.Linear(hidden_dim,len(vocabulary)) 

        #Dropout layer is added for regularization to prevent overfitting by randomly setting a fraction of the input units to 0 during training.
        self.dropout=nn.Dropout(dropout)
        
    
    def forward(self,x):
        """
        Forward pass of the model.
        :param x: Input tensor of shape (batch_size, sequence_length) containing word indices representing the current words in the articles.
        :return: Output tensor of shape (batch_size, sequence_length, vocabulary_size) containing the predicted probabilities for the next word in the sequence for each position in the input sequence.
        """
        # Pass input through embedding layer
        embedded = self.embedding(x) 

        #ignore the second output of the LSTM layer which contains the hidden and cell 
        #(batch_size, sequence_length, hidden_dim)
        lstm_out, _ = self.lstm(embedded) 

        # Apply dropout to the output of the LSTM layer
        lstm_out = self.dropout(lstm_out) 

        # Take the output of the last time step and pass it through the fully connected layer
        #(batch_size, sequence_length, vocabulary_size)
        logits = self.fc(lstm_out) 

        return logits
    

