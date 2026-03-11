from testing import testing_model
from training import training_model
from vocabulary import Vocabulary
from WordDataset import WordDataset, collate_fn
from NextWordPredictor import NextWordPredictor
import auxfunctions as aux

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#Routes for the datasets
test_route="data/test.parquet"
train_route="data/train.parquet"
validation_route="data/validation.parquet"

#Hyperparameters:
sequence_length = 256  # Reducido (menos contexto, más ejemplos)
embedding_dim = 512    # Reducido
hidden_dim = 1024       # Reducido
num_layers = 6         # ¡Solo 3 capas! (era 6)
dropout = 0.4         # ¡Más dropout! (era 0.2)
batch_size = 16      # Aumentado (más estable)
learning_rate = 0.0003 # Menor learning rate
nhead = 4              # Bien así
chunksize = 1000       # Bien
num_epochs = 100 

#Optimizations:
torch.backends.cudnn.benchmark = True          # Optimiza convoluciones/transformers
torch.backends.cudnn.deterministic = False     # Permite optimizaciones no deterministas
torch.backends.cuda.matmul.allow_tf32 = True   # Activa tensor cores para matrices
torch.backends.cudnn.allow_tf32 = True         # Activa tensor cores para cudnn
torch.set_float32_matmul_precision('high')     # Usa tensor cores para FP32

#Datasets and dataloaders:
vocabulary=Vocabulary(dataframe=train_route)

train_dataset=WordDataset(lambda: aux.build_articles_dataframe(train_route, chunksize=chunksize), vocabulary, sequence_length=sequence_length,chunksize=chunksize)
validation_dataset=WordDataset(lambda: aux.build_articles_dataframe(validation_route, chunksize=chunksize), vocabulary, sequence_length=sequence_length,chunksize=chunksize)
test_dataset=WordDataset(lambda: aux.build_articles_dataframe(test_route, chunksize=chunksize), vocabulary, sequence_length=sequence_length,chunksize=chunksize)

train_dataloader=DataLoader(train_dataset, batch_size=batch_size,collate_fn=lambda batch: collate_fn(vocabulary,batch),num_workers=8,pin_memory=True,prefetch_factor=4)
validation_dataloader=DataLoader(validation_dataset, batch_size=batch_size,collate_fn=lambda batch: collate_fn(vocabulary,batch),num_workers=8,pin_memory=True,prefetch_factor=4)
test_dataloader=DataLoader(test_dataset, batch_size=batch_size,collate_fn=lambda batch: collate_fn(vocabulary,batch),num_workers=8,pin_memory=True,prefetch_factor=4)

#Model, Loss Function and Optimizer:
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=NextWordPredictor(vocabulary, embedding_dim=embedding_dim,sequence_length=sequence_length, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,nhead=nhead).to(device)
criterion=nn.CrossEntropyLoss(ignore_index=vocabulary.pad_index)

optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate)

#Training and Testing:
training_model(model, train_dataloader, validation_dataloader, criterion, optimizer, device=device, num_epochs=num_epochs)

# Load best model checkpoint for final test evaluation
best_model_checkpoint = torch.load('best_model.pth', weights_only=False)
checkpoint_vocabulary = best_model_checkpoint['vocabulary']
checkpoint_config = best_model_checkpoint['config']

# Transformer
model = NextWordPredictor(
    vocabulary=checkpoint_vocabulary,
    embedding_dim=checkpoint_config['embedding_dim'],
    hidden_dim=checkpoint_config['hidden_dim'],
    num_layers=checkpoint_config['num_layers'],
    dropout=checkpoint_config['dropout'],
    sequence_length=checkpoint_config['sequence_length'],
    nhead=checkpoint_config['nhead']
)

# Weight load
model.load_state_dict(best_model_checkpoint['model_state_dict'])
model = model.to(device)
model.eval()


#Test the best model on the test set
testing_model(model, test_dataloader, criterion, device=device)

                            