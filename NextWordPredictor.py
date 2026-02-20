
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
        self.sequence_length=sequence_length
    
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
    def generate(self,prompt,max_length=128,temperature=0.8):
        """
        :param prompt: Input string to be used as the initial context for generating the next word in the sequence.
        :param max_length: Maximum length of the generated sequence, which determines how many words will be generated after the initial prompt.
        :param temperature: Temperature parameter for controlling the randomness of the generated text. A higher temperature will result in more random and diverse output, while a lower temperature will make the output more deterministic and focused on the most likely next words.
        :return: Generated sequence of text, which includes the initial prompt followed by the predicted next words up to the specified maximum length.
        """

        sequence=self.vocabulary.text_to_sequence(prompt)

        #Add BOS token
        context=[self.vocabulary.bos_index]+sequence
        generated_tokens=[]
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):


                if(len(context)>self.sequence_length):
                    input_window=context[-self.sequence_length:]
                else:
                    input_window=context       
                
                input_tensor = torch.tensor([input_window]).to(self.device)
                logits = self.forward(input_tensor)

                #Get the predicted next word by taking the argmax of the output logits for the last time step and dividing by the temperature to control the randomness of the generated text. The logits are 
                # divided by the temperature parameter to adjust the probabilities of the next word predictions, where a higher temperature will make the 
                # distribution more uniform and a lower temperature will make it more peaked around the most likely next words.
                next_word_logits = logits[0, -1, :] / temperature  

            
                #Apply softmax to convert logits to probabilities, then sample from the distribution to get the next word index
                probs = torch.softmax(next_word_logits, dim=-1)

                #We set the probabilities of the padding token, BOS token, and UNK token to 0 to prevent the model from generating these tokens as the next word in the sequence, which helps to ensure that the generated text is more coherent and meaningful.
                probs[self.vocabulary.pad_index] = 0
                probs[self.vocabulary.bos_index] = 0
                probs[self.vocabulary.unk_index] = 0

                probs=probs/probs.sum()  #Re-normalize the probabilities after setting certain tokens to 0 to ensure that they sum to 1, which is necessary for sampling the next word from the probability distribution.

                #torch.multinomial is used to sample from the probability distribution of the next word, which allows for more diverse and creative text generation compared to always choosing the most likely next word (argmax). By sampling from the distribution, the model can generate different outputs each time, even with the same input prompt, which can lead to more interesting and varied generated text.
                next_word_index = torch.multinomial(probs, 1).item()
                
                if next_word_index==self.vocabulary.eos_index:
                    break

                context.append(next_word_index)
                generated_tokens.append(next_word_index)
        return prompt+" "+self.vocabulary.sequence_to_text(generated_tokens)