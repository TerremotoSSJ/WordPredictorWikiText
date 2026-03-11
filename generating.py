import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint load
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
model = model.to(device)
model.eval()

# Generate text based on a prompt
prompt = "As with previous Valkyira Chronicles games"
# Convert the prompt to lowercase to match the training data preprocessing, which can help improve the model's ability to generate coherent text based on the learned patterns from the training data.
prompt = prompt.lower()  
generated_text = model.generate(prompt, max_length=80, temperature=0.8, topk=50, repetition_penalty=1.2)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")