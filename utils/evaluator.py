import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, hamming_loss

# loss function: Binary Cross Entropy
def loss_fn(outputs, targets):
    """
    returns the binary cross entropy loss between the predicted outputs and the targets
    """
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)



# Training of the model for one epoch
def train_model(training_loader, model, optimizer = None, device = 'cuda', loss_fn = loss_fn):
    """
    trains the model for one epoch
    :param training_loader: training data loader
    :param model: model to be trained
    :param optimizer: optimizer
    :param device: device to be used for training
    :return: trained model, model accuracy, mean loss
    """
    losses = []
    correct_predictions = 0
    num_samples = 0
    total_batches = len(training_loader)

    # set model to training mode (activate dropout, batch norm)
    model.train()

    for batch_idx, data in enumerate(training_loader):
        ids = data['input_ids'].to(device, dtype=torch.int)
        mask = data['attention_mask'].to(device, dtype=torch.int)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.int)
        targets = data['targets'].to(device, dtype=torch.float)

        # forward
        outputs = model(ids, mask, token_type_ids)  # (batch,predict)=(32,8)
        # print(outputs)
        loss = loss_fn(outputs, targets)
        # print(loss)
        losses.append(loss.item())
        # training accuracy, apply sigmoid, round (apply thresh 0.5)
        outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
        targets = targets.cpu().detach().numpy()
        correct_predictions += np.sum(outputs == targets)
        # print(correct_predictions)
        num_samples += targets.size  # total number of elements in the 2D array

        # backward
        if optimizer: optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # grad descent step
        if optimizer: optimizer.step()

        # Print progress
        # print(f"Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item()}")

    # returning: trained model, model accuracy, mean loss
    return model, float(correct_predictions) / num_samples, np.mean(losses)



def evaluate_model(validation_loader, model, loss_fn = loss_fn, device = 'cuda', THRESHOLD = 0.5, target_list = []):
    model.eval()
    final_targets = []
    final_outputs = []
    losses = []

    with torch.no_grad():
        for data in validation_loader:
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            
            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            outputs = torch.sigmoid(outputs).cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            final_outputs.extend(outputs)
            final_targets.extend(targets)
    
    final_outputs = np.array(final_outputs) >= THRESHOLD
    # Calculating metrics
    acc = accuracy_score(final_targets, final_outputs)
    f1 = f1_score(final_targets, final_outputs, average='micro')  # Consider using 'macro' or 'weighted' based on your problem
    precision = precision_score(final_targets, final_outputs, average='micro')
    recall = recall_score(final_targets, final_outputs, average='micro')
    hamming = hamming_loss(final_targets, final_outputs)
    
    average_loss = np.mean(losses)
    
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Hamming Loss: {hamming}")
    print(f"Average Loss: {average_loss}")
    # Detailed classification report
    print("\nClassification Report:\n", classification_report(final_targets, final_outputs, target_names=target_list))

    return acc, f1, precision, recall, hamming, average_loss