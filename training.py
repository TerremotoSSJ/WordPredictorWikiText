import torch
from torch.utils.data import DataLoader, IterableDataset

from vocabulary import Vocabulary
import auxfunctions as aux
from WordDataset import WordDataset
from NextWordPredictor import NextWordPredictor

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_route="data/train.parquet"
validation_route="data/validation.parquet"

train_vocabulary=Vocabulary(dataframe=train_route)


train_dataset=WordDataset(aux.build_articles_dataframe(train_route), train_vocabulary, minimum_sequence_length=4, maximum_sequence_length=64, step=1)
validation_dataset=WordDataset(aux.build_articles_dataframe(validation_route), train_vocabulary, minimum_sequence_length=4, maximum_sequence_length=64, step=1)

train_dataloader=DataLoader(train_dataset, batch_size=64,collate_fn=lambda batch: aux.collate_fn(train_vocabulary,batch))
validation_dataloader=DataLoader(validation_dataset, batch_size=64,collate_fn=lambda batch: aux.collate_fn(train_vocabulary,batch))

for batch in train_dataloader:
    print(batch)
    break