
import torch
import torch.nn as nn
import torch.optim as optim

def testing_model(model, test_dataloader, criterion,device=None):
    """
    Docstring for testing_model

    :param model: The NextWordPredictor model to be tested.
    :param test_dataloader: DataLoader for the test dataset, which provides batches of preprocessed articles and their corresponding next word targets for evaluating the model's performance on unseen data after training.
    :param criterion: Loss function used to compute the loss between the model's predictions and the actual next word targets during testing.
    :param device: Device on which to perform the testing, which can be either "cuda" for GPU or "cpu" for CPU. If not specified, the function will automatically use GPU if available, otherwise it will default to CPU.
    :return: Average test loss computed over the entire test dataset, which indicates the model's performance on unseen data after training.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_test_loss = 0
    num_test_batches = 0
    with torch.no_grad():
        for current_word_targets, next_word_targets,attention_mask, lengths_current,lengths_next in test_dataloader:
            current_word_targets = current_word_targets.to(device)
            next_word_targets = next_word_targets.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(current_word_targets, attention_mask=attention_mask)
            test_loss = criterion(outputs.permute(0, 2, 1), next_word_targets)
            total_test_loss += test_loss.item()
            num_test_batches += 1
            print(f"Test Batch {num_test_batches}, Loss: {test_loss.item():.4f}")

    avg_test_loss = total_test_loss / num_test_batches if num_test_batches > 0 else 0
    print(f"Test Loss: {avg_test_loss:.4f}")