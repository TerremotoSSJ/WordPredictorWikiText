
from multiprocessing import context

from sympy import sequence

from vocabulary import Vocabulary
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class NextWordPredictor(nn.Module):
    """
    Transformer model for next word prediction. 
    """

    def __init__(self, vocabulary, embedding_dim, hidden_dim, num_layers=5,dropout=0.2,sequence_length=258,nhead=8,device=None):

        super(NextWordPredictor, self).__init__()
        self.vocabulary = vocabulary
        self.embedding = nn.Embedding(len(vocabulary), embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead=nhead
        self.embedding_dim=embedding_dim
        #Multi-layer Transformer
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout,batch_first=True,activation='relu')
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        #output layer 
        self.fc = nn.Linear(embedding_dim, len(vocabulary)) 
        self.sequence_length = sequence_length

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #Positional encoding is added
        self.positional_encoding=nn.Parameter(torch.randn(1, sequence_length, embedding_dim))
    
    def forward(self,x,attention_mask):
        """
        Forward pass of the model.
        :param x: Input tensor of shape (batch_size, sequence_length) containing word indices representing the current words in the articles.
        :param attention: Attention mask tensor of shape (batch_size, sequence_length) indicating which positions in the input sequence should be attended to (1 for valid positions, 0 for padding).
        :return: Output tensor of shape (batch_size, sequence_length, vocabulary_size) containing the predicted probabilities for the next word in the sequence for each position in the input sequence.
        """
        # Pass input through embedding layer
        embedded = self.embedding(x) 
        transformer_input=embedded+self.positional_encoding[:, :embedded.size(1), :]
        src_attention_mask=None
        #attention mask
        if attention_mask is not None:
            src_attention_mask=(attention_mask==0)
        
        #causal mask to prevent the model from attending to future positions in the sequence, which ensures that the prediction for the next word at each position is based only on the current and previous words in the sequence, thus maintaining the autoregressive nature of the language model.
        seq_len = x.size(1)
        causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), 
        diagonal=1
    )
        
        transformer_output = self.transformer(transformer_input, src_key_padding_mask=src_attention_mask, mask=causal_mask)
        transformer_output = self.dropout(transformer_output)
        logits = self.fc(transformer_output) 

        return logits
    def generate(self, prompt, max_length=128, temperature=0.8, topk=50):
        """
    :param prompt: Input string to start generation
    :param max_length: Maximum length of generated sequence
    :param temperature: Randomness control
    :param topk: Number of top tokens to consider (0 = disabled)
    :return: Generated text
    """
        print(f"Generating with prompt: '{prompt}'")
    
        sequence = self.vocabulary.text_to_sequence(prompt)
        print(f"Sequence: {sequence}")
    
        context = [self.vocabulary.bos_index] + sequence
        print(f"Context with BOS: {context}")
    
        generated_tokens = []
    
        self.eval()
        with torch.no_grad():
            for step in range(max_length):
                # Handle context window
                if len(context) > self.sequence_length:
                    input_window = context[-self.sequence_length:]
                else:
                    input_window = context       
            
                print(f"Step {step}, input_window length: {len(input_window)}")
            
                input_tensor = torch.tensor([input_window]).to(self.device)
                attention_mask = torch.ones(1, len(input_window)).to(self.device)  
                logits = self.forward(input_tensor, attention_mask=attention_mask)
            
                next_word_logits = logits[0, -1, :] / temperature  
                probs = torch.softmax(next_word_logits, dim=-1)
            
                # Top-k filtering
                k = min(topk, len(probs))
                print(f"Top-k={k}")
                values, indices = torch.topk(probs, k=k)
                new_prob = torch.zeros_like(probs)
                new_prob.scatter_(0, indices, values)
            
                # Prohibit special tokens
                new_prob[self.vocabulary.pad_index] = 0
                new_prob[self.vocabulary.bos_index] = 0
                new_prob[self.vocabulary.unk_index] = 0
            
                # Check if all probabilities are zero
                if new_prob.sum() == 0:
                    print("⚠️ All probabilities are zero! Cannot sample.")
                    break
            
                new_prob = new_prob / new_prob.sum()
            
                next_word_index = torch.multinomial(new_prob, 1).item()
                print(f"Generated token index: {next_word_index}")
            
                if next_word_index == self.vocabulary.eos_index and len(generated_tokens) > 10:
                    break  
                if next_word_index == self.vocabulary.eos_index and len(generated_tokens) <= 10:
                    continue
            
                context.append(next_word_index)
                generated_tokens.append(next_word_index)
            
                # Print partial result
                partial = self.vocabulary.sequence_to_text(generated_tokens)
                print(f"  → {partial}")
    
        final_text = prompt + " " + self.vocabulary.sequence_to_text(generated_tokens)
        print(f"Final: '{final_text}'")
        return final_text