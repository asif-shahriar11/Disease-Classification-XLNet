from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import AdamW

from torch.nn import BCEWithLogitsLoss

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, hamming_loss


# Hyperparameters
MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-05
THRESHOLD = 0.5 # threshold for the sigmoid


tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        self.title = list(df['File Contents'])
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'title': title
        }
    

train_df = pd.read_csv('train_csv.csv')
test_df = pd.read_csv('test_csv.csv')
# split test into test and validation datasets
train_df, val_df = train_test_split(train_df, random_state=88, test_size=0.30, shuffle=True)

target_list = list(train_df.columns[1:])

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


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# XLNET model
class XLNETBase(nn.Module):
    def __init__(self):
        super(XLNETBase, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 23) # 23 is the number of classes in ohsumed dataset

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output = self.drop(pooled_output)
        return self.out(output)

model = XLNETBase()
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# define the optimizer
optimizer = AdamW(model.parameters(), lr = LEARNING_RATE)


# Training of the model for one epoch
def train_model(training_loader, model, optimizer):
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
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # grad descent step
        optimizer.step()

        # Print progress
        # print(f"Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item()}")

    # returning: trained model, model accuracy, mean loss
    return model, float(correct_predictions) / num_samples, np.mean(losses)




def eval_model(validation_loader, model):
    model.eval()
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for data in validation_loader:
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
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
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Hamming Loss: {hamming}")
    # Detailed classification report
    print("\nClassification Report:\n", classification_report(final_targets, final_outputs, target_names=target_list))


    # losses = []
    # correct_predictions = 0
    # num_samples = 0
    # # set model to eval mode (turn off dropout, fix batch norm)
    # model.eval()

    # with torch.no_grad():
    #     for batch_idx, data in enumerate(validation_loader, 0):
    #         ids = data['input_ids'].to(device, dtype = torch.long)
    #         mask = data['attention_mask'].to(device, dtype = torch.long)
    #         token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
    #         targets = data['targets'].to(device, dtype = torch.float)
    #         outputs = model(ids, mask, token_type_ids)

    #         loss = loss_fn(outputs, targets)
    #         losses.append(loss.item())

    #         # validation accuracy
    #         # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
    #         outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
    #         targets = targets.cpu().detach().numpy()
    #         correct_predictions += np.sum(outputs==targets)
    #         num_samples += targets.size   # total number of elements in the 2D array

    # return float(correct_predictions)/num_samples, np.mean(losses)



# -------------------------- For Training -----------------------------------


# history = defaultdict(list)
# best_accuracy = 0

# for epoch in range(1, EPOCHS+1):
#     print(f'Epoch {epoch}/{EPOCHS}')
#     model, train_acc, train_loss = train_model(train_data_loader, model, optimizer)
#     eval_model(val_data_loader, model)

#     history['train_acc'].append(train_acc)
#     history['train_loss'].append(train_loss)
#     history['val_acc'].append(val_acc)
#     history['val_loss'].append(val_loss)
#     # save the best model
#     if val_acc > best_accuracy:
#         torch.save(model.state_dict(), "XLNET_MLTC_model_state.bin")
#         best_accuracy = val_acc

# -------------------------- For Training -----------------------------------


# -------------------------- For Test -----------------------------------

# Loading pretrained model (best model)
# model = XLNETBase()
# model.load_state_dict(torch.load("XLNET_MLTC_model_state.bin"))
# model = model.to(device)


# # Evaluate the model using the test data
# eval_model(test_data_loader, model)
# test_acc, test_loss = eval_model(test_data_loader, model, optimizer)
# print(f'test loss: {test_loss}, test acc: {test_acc}')

# -------------------------- For Test -----------------------------------