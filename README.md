# WordPredictorWikiText

Next word transformer word predictor using WikiText-2 and Pytorch

### Model Info:
- 6-layer Transformer Model
- Causal Mask and Attention Mask
- Using sharding for multiple workers in IterableDataset
- <PAD> <UNK> <EOS> <BOS> tokens usage
- Gradient Clipping
- Early Stopping
- Temperature
- Top k sampling
- Generating predicted next words given an input prompt
- Saving Checkpoints 

## Installation
```bash
git clone -b v1.1 https://github.com/TerremotoSSJ/WordPredictorWikiText.git
cd WordPredictorWikiText
pip install -r requirements.txt
```

## Training

The model will start training using:
```bash
python main.py
```

## Generating text
```python
import torch
from NextWordPredictor import NextWordPredictor

best_model_checkpoint = torch.load('best_model.pth', weights_only=False)

vocabulary = best_model_checkpoint['vocabulary']
embedding_dim = best_model_checkpoint['config']['embedding_dim']
hidden_dim = best_model_checkpoint['config']['hidden_dim']
num_layers = best_model_checkpoint['config']['num_layers']
dropout = best_model_checkpoint['config']['dropout']
sequence_length = best_model_checkpoint['config']['sequence_length']
nhead = best_model_checkpoint['config']['nhead']

# Transformer
from NextWordPredictor import NextWordPredictor
model = NextWordPredictor(
    vocabulary=vocabulary,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    sequence_length=sequence_length,
    nhead=nhead
)

# Weight load
model.load_state_dict(best_model_checkpoint['model_state_dict'])

prompt = "Input prompt"
generated = model.generate(prompt, max_length=50)
print(generated)
```

## Hyperparameters (default)

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

## Proyect structure

```
├── NextWordPredictor.py      # Transformer Model
├── WordDataset.py            # Dataset and collate_fn
├── vocabulary.py             # Special tokens and Vocabulary
├── auxfunctions.py           # Preprocessing data
├── training.py               # Training Loop
├── testing.py                # Testing the model
├── main.py                   # Training and testing model
├── generating.py             # Generate text
└── best_model.pth            # Trained Model