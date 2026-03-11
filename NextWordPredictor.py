
import math

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
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout,batch_first=True,activation='relu',norm_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers, enable_nested_tensor=False)
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
        embedded = self.embedding(x)* math.sqrt(self.embedding_dim)  
        transformer_input=embedded+self.positional_encoding[:, :embedded.size(1), :]
        src_attention_mask=None
        #padding mask
        if attention_mask is not None:
            src_attention_mask=(attention_mask==0)
        
        #causal mask to prevent the model from attending to future positions in the sequence, which ensures that the prediction for the next word at each position is based only on the current and previous words in the sequence, thus maintaining the autoregressive nature of the language model.
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        
        transformer_output = self.transformer(transformer_input, src_key_padding_mask=src_attention_mask, mask=causal_mask)
        transformer_output = self.dropout(transformer_output)
        logits = self.fc(transformer_output) 

        return logits
    def generate(self, prompt, max_length=128, temperature=0.8, topk=50, repetition_penalty=1.2):
        """
    :param prompt: Input string to start generation
    :param max_length: Maximum length of generated sequence
    :param temperature: Randomness control
    :param topk: Number of top tokens to consider (0 = disabled)
    :return: Generated text
    """
        if(topk==0):
            topk = len(self.vocabulary)
        sequence = self.vocabulary.text_to_sequence(prompt)

    
        context = [self.vocabulary.bos_index] + sequence

        generated_tokens = []
    
        self.eval()
        with torch.no_grad():
            model_device = next(self.parameters()).device
            for step in range(max_length):
                # Handle context window
                if len(context) > self.sequence_length:
                    input_window = context[-self.sequence_length:]
                else:
                    input_window = context       
            
            
                input_tensor = torch.tensor([input_window], device=model_device)
                attention_mask = torch.ones(1, len(input_window), device=model_device)
                logits = self.forward(input_tensor, attention_mask=attention_mask)
            
                next_word_logits = logits[0, -1, :] / temperature

                # Penalize already generated tokens to reduce loops/repetitions.
                if repetition_penalty > 1.0 and len(generated_tokens) > 0:
                    repeated = torch.tensor(list(set(generated_tokens)), device=model_device)
                    next_word_logits[repeated] = next_word_logits[repeated] / repetition_penalty

                probs = torch.softmax(next_word_logits, dim=-1)
            
                # Top-k filtering
                k = min(topk, len(probs))
                values, indices = torch.topk(probs, k=k)
                new_prob = torch.zeros_like(probs)
                new_prob.scatter_(0, indices, values)
            
                # Prohibit special tokens
                new_prob[self.vocabulary.pad_index] = 0
                new_prob[self.vocabulary.bos_index] = 0
                new_prob[self.vocabulary.unk_index] = 0
                if(new_prob.sum() == 0):
                    probs[self.vocabulary.bos_index] = 0
                    probs[self.vocabulary.unk_index] = 0
                    probs[self.vocabulary.pad_index] = 0
                    probs = probs / probs.sum()
                    new_prob=probs
                else:
                    new_prob = new_prob / new_prob.sum()
            
                next_word_index = torch.multinomial(new_prob, 1).item()
            
                if next_word_index == self.vocabulary.eos_index:
                    if len(generated_tokens) >= 20:  # Ensure at least 20 tokens are generated before allowing EOS
                        break
                    else:
                        continue  # Retry if EOS is generated 

                context.append(next_word_index)
                generated_tokens.append(next_word_index)
            
                # Print partial result
                partial = self.vocabulary.sequence_to_text(generated_tokens)
    
        final_text = prompt + " " + self.vocabulary.sequence_to_text(generated_tokens)
        return final_text