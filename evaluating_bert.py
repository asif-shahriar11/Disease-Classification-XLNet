import numpy as np 
import pandas as pd
import torch
import torch.nn as nn

from collections import defaultdict
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split

from utils.evaluator import train_model, evaluate_model
from model.custom_dataset import CustomDataset
from model.bert import BERTClass


# Hyperparameters
MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-05
THRESHOLD = 0.5 # threshold for the sigmoid

# model
model = BERTClass()

# tokenizer
tokenizer = model.get_tokenizer()

# data
train_df = pd.read_csv('train_csv.csv')
test_df = pd.read_csv('test_csv.csv')
# split test into test and validation datasets
train_df, val_df = train_test_split(train_df, random_state=88, test_size=0.30, shuffle=True)

target_list = list(train_df.columns[1:])
# print(target_list)

train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN, target_list)
valid_dataset = CustomDataset(val_df, tokenizer, MAX_LEN, target_list)
test_dataset = CustomDataset(test_df, tokenizer, MAX_LEN, target_list)

# print(train_dataset[0])

# Data loaders
train_data_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_data_loader = torch.utils.data.DataLoader(test_dataset, 
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device) # move model to device

# loss function
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# optimizer
optimizer = AdamW(model.parameters(), lr = LEARNING_RATE)


# Evaluate the model using the validation data

history = defaultdict(list)
best_accuracy = 0
best_f1 = 0

for epoch in range(1, EPOCHS+1):
    print(f'Epoch {epoch}/{EPOCHS}')
    model, train_acc, train_loss = train_model(train_data_loader, model, optimizer)
    #val_acc, val_loss = evaluate_model(val_data_loader, model, optimizer)
    val_acc, val_f1, val_precision, val_recall, val_hamming_loss, val_loss = evaluate_model(val_data_loader, model, loss_fn, device, THRESHOLD, target_list)

    print(f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f} train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    history['val_precision'].append(val_precision)
    history['val_recall'].append(val_recall)
    history['val_loss'].append(val_loss)
    # save the best model based on accuracy
    # if val_acc > best_accuracy:
    #     torch.save(model.state_dict(), "BERT_MLTC_model_state.bin")
    #     best_accuracy = val_acc
    # save the best model based on f1 score
    if val_f1 > best_f1:
        torch.save(model.state_dict(), "BERT_MLTC_model_state.bin")
        best_f1 = val_f1




# Test the model

# Loading pretrained model (best model)
model = BERTClass()
model.load_state_dict(torch.load("BERT_MLTC_model_state.bin"))
model = model.to(device)


# Evaluate the model using the test data
#test_acc, test_loss = evaluate_model(test_data_loader, model, optimizer)
evaluate_model(test_data_loader, model, loss_fn, device, THRESHOLD, target_list)
