from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from WordDataset import collate_fn
from vocabulary import Vocabulary
import auxfunctions as aux
from WordDataset import WordDataset
from NextWordPredictor import NextWordPredictor



def training_model(model, train_dataloader, validation_dataloader, criterion, optimizer,device=None, num_epochs=100):
    """
    Docstring for training_model

    :param model: The NextWordPredictor model to be trained.
    :param train_dataloader: DataLoader for the training dataset, which provides batches of preprocessed articles and their corresponding next word targets for training the model.
    :param validation_dataloader: DataLoader for the validation dataset, which provides batches of preprocessed articles and their corresponding next word targets for evaluating the model's performance on unseen data during training.
    :param criterion: Loss function used to compute the loss between the model's predictions and the actual next word targets during training.
    :param optimizer: Optimization algorithm used to update the model's parameters based on the computed loss during training.
    :param device: Device on which to perform the training, which can be either "cuda" for GPU or "cpu" for CPU. If not specified, the function will automatically use GPU if available, otherwise it will default to CPU.
    :param num_epochs: Number of epochs to train the model, where each epoch represents one complete pass through the entire training dataset. The default value is set to 100 epochs, but it can be adjusted based on the specific requirements of the training process and the size of the dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_loss=float('inf')
    not_improving_epochs=0
    for epoch in range(num_epochs):
        model.train()
        total_loss=0

        #The dataset is a streaming iterable dataset so we can not use len(train_dataloader) to compute the average loss, instead we keep track of the number of batches processed and divide the total loss by that number at the end of the epoch to get the average loss per batch.
        num_batches=0
        for batch_idx, (current_word_targets, next_word_targets, lengths_current,lengths_next) in enumerate(train_dataloader):
            current_word_targets=current_word_targets.to(device)
            next_word_targets=next_word_targets.to(device)

            optimizer.zero_grad()
            outputs=model(current_word_targets)
            loss = criterion(
                outputs.permute(0, 2, 1),
                next_word_targets         
)
            loss.backward()

            #Gradient clipping is a technique used to prevent the exploding gradient problem during training, where the gradients can become excessively large and cause instability in the training process. By setting a maximum norm for the gradients, we can ensure that they do not exceed a certain threshold, which helps maintain stable training and prevents the model from diverging.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue
            optimizer.step()

            total_loss+=loss.item()
            num_batches += 1
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
        
        avg_train_loss=total_loss/num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation Loop
        model.eval()
        total_val_loss=0
        num_val_batches=0
        with torch.no_grad():
            for current_word_targets, next_word_targets, lengths_current,lengths_next in validation_dataloader:
                current_word_targets=current_word_targets.to(device)
                next_word_targets=next_word_targets.to(device)

                outputs=model(current_word_targets)
                val_loss=criterion(outputs.permute(0, 2, 1), next_word_targets)
                num_val_batches += 1
                total_val_loss+=val_loss.item()
                print(f"Epoch {epoch+1}, Validation Batch {num_val_batches}, Loss: {val_loss.item():.4f}")
        
        avg_val_loss=total_val_loss/num_val_batches if num_val_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            #Save the best model checkpoint 
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocabulary': model.vocabulary,  
                'best_val_loss': best_val_loss,
                'config': {
                    'embedding_dim': model.embedding.embedding_dim,
                    'hidden_dim': model.lstm.hidden_size,
                    'num_layers': model.lstm.num_layers,
                    'dropout': model.dropout.p,
                    'sequence_length': model.sequence_length
                }
            }
            torch.save(checkpoint, "best_model.pth")
        else:
            not_improving_epochs += 1
            if not_improving_epochs >= 3:
                print("Early stopping triggered. No improvement in validation loss for 3 consecutive epochs.")
                break