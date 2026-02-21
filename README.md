# WordPredictorWikiText

Next word LSTM word predictor using WikiText-2 and Pytorch

### Model Info:
- 3-layer LSTM model
- Training using BPTT (Backpropagation through time)
- Using sharding for multiple workers in IterableDataset
- <PAD> <UNK> <EOS> <BOS> tokens usage
- Gradient Clipping
- Early Stopping
- Temperature
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

checkpoint = torch.load('best_model.pth', map_location='cpu', weights_only=False)
model = NextWordPredictor(
    vocabulary=checkpoint['vocabulary'],
    embedding_dim=checkpoint['config']['embedding_dim'],
    hidden_dim=checkpoint['config']['hidden_dim'],
    num_layers=checkpoint['config']['num_layers'],
    dropout=checkpoint['config']['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

prompt = "Input prompt"
generated = model.generate(prompt, max_length=50)
print(generated)
```

## Hyperparameters (default)

embedding_dim=400
hidden_dim=1024
num_layers=3
dropout=0.4
sequence_length=128
batch_size=64
learning_rate=0.001
max_norm=1

## Proyect structure

```
├── NextWordPredictor.py      # LSTM Model
├── WordDataset.py            # Dataset and collate_fn
├── vocabulary.py             # Special tokens and Vocabulary
├── auxfunctions.py           # Preprocessing data
├── training.py               # Training Loop
├── testing.py                # Testing the model
├── main.py                   # Training and testing model
├── generating.py             # Generate text
└── best_model.pth            # Trained Model