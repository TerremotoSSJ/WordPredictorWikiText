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

#Datasets and dataloaders:
vocabulary=Vocabulary(dataframe=train_route)

train_dataset=WordDataset(aux.build_articles_dataframe(train_route), vocabulary, minimum_sequence_length=4, maximum_sequence_length=128, step=1)
validation_dataset=WordDataset(aux.build_articles_dataframe(validation_route), vocabulary, minimum_sequence_length=4, maximum_sequence_length=128, step=1)
test_dataset=WordDataset(aux.build_articles_dataframe(test_route), vocabulary, minimum_sequence_length=4, maximum_sequence_length=128, step=1)

train_dataloader=DataLoader(train_dataset, batch_size=256,collate_fn=lambda batch: collate_fn(vocabulary,batch),num_workers=4,pin_memory=True)
validation_dataloader=DataLoader(validation_dataset, batch_size=256,collate_fn=lambda batch: collate_fn(vocabulary,batch),num_workers=4,pin_memory=True)
test_dataloader=DataLoader(test_dataset, batch_size=256,collate_fn=lambda batch: collate_fn(vocabulary,batch),num_workers=4,pin_memory=True)

#Model, Loss Function and Optimizer:
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=NextWordPredictor(vocabulary, embedding_dim=400, hidden_dim=1024, num_layers=3, dropout=0.4).to(device)
criterion=nn.CrossEntropyLoss(ignore_index=vocabulary.pad_index)
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)    

#Training and Testing:
training_model(model, train_dataloader, validation_dataloader, criterion, optimizer, device=device, num_epochs=100)
testing_model(model, test_dataloader, criterion, device=device)

                            