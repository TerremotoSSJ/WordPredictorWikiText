
from vocabulary import Vocabulary
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class NextWordPredictor(nn.Module):
    """
    LSTM-based model for next word prediction in a sequence of text.
    """

    def __init__(self, vocabulary, embedding_dim, hidden_dim, num_layers=4,dropout=0.2,sequence_length=128):
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

        self.vocabulary=vocabulary
        self.device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.length=sequence_length
    
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
    def generate(self,prompt,max_length=128):
        """
        :param prompt: Input string to be used as the initial context for generating the next word in the sequence.
        :param max_length: Maximum length of the generated sequence, which determines how many words will be generated after the initial prompt.
        :return: Generated sequence of text, which includes the initial prompt followed by the predicted next words up to the specified maximum length.
        """

        sequence=self.vocabulary.text_to_sequence(prompt)

        #Add BOS token
        sequence=[self.vocabulary.bos_index]+sequence

        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                if(len(sequence)>self.length):
                    input_window=sequence[-self.length:]
                else:
                    input_window=sequence 
                input_seq=torch.tensor([input_window]).to(self.device)
                logits=self.forward(input_window) 

                #Get the predicted next word by taking the argmax of the output logits for the last time step
                next_word_logits=logits[0, -1, :] 
                next_word_index=torch.argmax(next_word_logits).item() 
                if next_word_index==self.vocabulary.eos_index:
                    break
                sequence.append(next_word_index)
        return self.vocabulary.sequence_to_text(sequence)