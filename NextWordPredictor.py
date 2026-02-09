
from vocabulario import Vocabulary
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class NextWordPredictor(nn.Module):
    """
    LSTM-based model for next word prediction in a sequence of text.
    """

    def __init__(self, vocabulary, embedding_dim, hidden_dim, num_layers=2):
        super(NextWordPredictor, self).__init__()

        #Embedding layer converts word indices into dense vectors of fixed size (embedding_dim).
        self.embedding = nn.Embedding(len(vocabulary), embedding_dim)

        #Dropout is added to prevent overfitting by randomly setting a fraction of the input units to 0 during training.
        #Batch_first=True ensures that the input and output tensors are of shape (batch_size, sequence_length, hidden_dim).
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,num_layers,batch_first=True,dropout=0.2) 
        self.fc=nn.Linear(hidden_dim,len(vocabulary)) 

        #Dropout layer is added for regularization to prevent overfitting by randomly setting a fraction of the input units to 0 during training.
        self.dropout=nn.Dropout(0.2)
        
    
    def forward(self,x):
        """
        Forward pass of the model.

        :param x: Input tensor of shape (batch_size, sequence_length) containing word indices.
        :return: Output tensor of shape (batch_size, vocab_size) containing the predicted probabilities for the next word.
        """
        # Pass input through embedding layer
        embedded = self.embedding(x) 

        #ignore the second output of the LSTM layer which contains the hidden and cell states
        lstm_out, _ = self.lstm(embedded) 

        # Apply dropout to the output of the LSTM layer
        lstm_out = self.dropout(lstm_out) 

        # Take the output of the last time step and pass it through the fully connected layer
        output = self.fc(lstm_out[:, -1, :]) 

        return output
    
newVocablary = Vocabulary()
text = "This is a sample text for testing the next word predictor."
text2= "Another example to test the model's ability to predict the next word."
sequence = newVocablary.text_to_sequence(text)
sequence2 = newVocablary.text_to_sequence(text2)
sequence1=newVocablary.pad_sequence(sequence, max_length=10)  # Pad the first sequence to a maximum length of 10
sequence2=newVocablary.pad_sequence(sequence2, max_length=10)  # Pad the second sequence to a maximum length of 10
print(sequence1)
test=NextWordPredictor(newVocablary, embedding_dim=50, hidden_dim=100)
input_tensor = torch.tensor([sequence1], dtype=torch.long)  # Convert sequence to tensor and
bothsequences= torch.tensor([sequence1, sequence2], dtype=torch.long)  # Convert both sequences to tensor
output = test(bothsequences)  # Pass both sequences through the model
print(output)
