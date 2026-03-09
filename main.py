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
sequence_length=256
embedding_dim=256
hidden_dim=512
num_layers=6
dropout=0.2
batch_size=16
num_epochs=100
learning_rate=0.001
nhead=4
chunksize=1000

#Datasets and dataloaders:
vocabulary=Vocabulary(dataframe=train_route)

train_dataset=WordDataset(aux.build_articles_dataframe(train_route), vocabulary, sequence_length=sequence_length,chunksize=chunksize)
validation_dataset=WordDataset(aux.build_articles_dataframe(validation_route), vocabulary, sequence_length=sequence_length,chunksize=chunksize)
test_dataset=WordDataset(aux.build_articles_dataframe(test_route), vocabulary, sequence_length=sequence_length,chunksize=chunksize)

train_dataloader=DataLoader(train_dataset, batch_size=batch_size,collate_fn=lambda batch: collate_fn(vocabulary,batch),num_workers=4,pin_memory=True)
validation_dataloader=DataLoader(validation_dataset, batch_size=batch_size,collate_fn=lambda batch: collate_fn(vocabulary,batch),num_workers=4,pin_memory=True)
test_dataloader=DataLoader(test_dataset, batch_size=batch_size,collate_fn=lambda batch: collate_fn(vocabulary,batch),num_workers=4,pin_memory=True)

#Model, Loss Function and Optimizer:
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=NextWordPredictor(vocabulary, embedding_dim=embedding_dim,sequence_length=sequence_length, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,nhead=nhead).to(device)
criterion=nn.CrossEntropyLoss(ignore_index=vocabulary.pad_index)

optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)    

#Training and Testing:
training_model(model, train_dataloader, validation_dataloader, criterion, optimizer, device=device, num_epochs=num_epochs)
testing_model(model, test_dataloader, criterion, device=device)

                            