
import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_checkpoint = torch.load('best_model.pth',weights_only=False)
model=best_model_checkpoint['model_state_dict']
vocabulary=best_model_checkpoint['vocabulary']
embedding_dim=best_model_checkpoint['config']['embedding_dim']
hidden_dim=best_model_checkpoint['config']['hidden_dim']
num_layers=best_model_checkpoint['config']['num_layers']
dropout=best_model_checkpoint['config']['dropout']

from NextWordPredictor import NextWordPredictor
model=NextWordPredictor(vocabulary, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
model.load_state_dict(best_model_checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
prompt="The experiment was a "
generated_text=model.generate(prompt, max_length=50)
print(generated_text)