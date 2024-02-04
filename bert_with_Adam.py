import numpy as np 
import pandas as pd
import torch
import torch.nn as nn

import tqdm.notebook as tq

from transformers import BertTokenizer, BertModel, AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_df = pd.read_csv('train_csv.csv')
test_df = pd.read_csv('test_csv.csv')

# Hyperparameters
MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 1e-05
THRESHOLD = 0.5 # threshold for the sigmoid


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

target_list = list(train_df.columns[1:])
# print(target_list)

train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN, target_list)
test_dataset = CustomDataset(test_df, tokenizer, MAX_LEN, target_list)

# print(train_dataset[0])

# Data loaders
train_data_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_data_loader = torch.utils.data.DataLoader(test_dataset, 
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# bert model
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.25)
        self.linear = torch.nn.Linear(768, 23)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

model = BERTClass()
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# define the optimizer
optimizer = AdamW(model.parameters(), lr = LEARNING_RATE)


# # Training of the model for one epoch
# def train_model(training_loader, model, optimizer):

#     losses = []
#     correct_predictions = 0
#     num_samples = 0
#     # set model to training mode (activate dropout, batch norm)
#     model.train()
#     # initialize the progress bar
#     loop = tq.tqdm(enumerate(training_loader), total=len(training_loader), 
#                       leave=True, colour='steelblue')
#     for batch_idx, data in loop:
#         ids = data['input_ids'].to(device, dtype = torch.int)
#         mask = data['attention_mask'].to(device, dtype = torch.int)
#         token_type_ids = data['token_type_ids'].to(device, dtype = torch.int)
#         targets = data['targets'].to(device, dtype = torch.float16)

#         # forward
#         outputs = model(ids, mask, token_type_ids) # (batch,predict)=(32,8)
#         loss = loss_fn(outputs, targets)
#         losses.append(loss.item())
#         # training accuracy, apply sigmoid, round (apply thresh 0.5)
#         outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
#         targets = targets.cpu().detach().numpy()
#         correct_predictions += np.sum(outputs==targets)
#         num_samples += targets.size   # total number of elements in the 2D array

#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         # grad descent step
#         optimizer.step()

#         # Update progress bar
#         #loop.set_description(f"")
#         #loop.set_postfix(batch_loss=loss)

#     # returning: trained model, model accuracy, mean loss
#     return model, float(correct_predictions)/num_samples, np.mean(losses)


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
        targets = data['targets'].to(device, dtype=torch.float16)

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
        print(f"Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item()}")

    # returning: trained model, model accuracy, mean loss
    return model, float(correct_predictions) / num_samples, np.mean(losses)


def eval_model(validation_loader, model, optimizer):
    losses = []
    correct_predictions = 0
    num_samples = 0
    # set model to eval mode (turn off dropout, fix batch norm)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            # validation accuracy
            # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()
            correct_predictions += np.sum(outputs==targets)
            num_samples += targets.size   # total number of elements in the 2D array

    return float(correct_predictions)/num_samples, np.mean(losses)


# history = defaultdict(list)
best_accuracy = 0

for epoch in range(1, EPOCHS+1):
    print(f'Epoch {epoch}/{EPOCHS}')
    model, train_acc, train_loss = train_model(train_data_loader, model, optimizer)

    print(f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}')

    # history['train_acc'].append(train_acc)
    # history['train_loss'].append(train_loss)
    # history['val_acc'].append(val_acc)
    # history['val_loss'].append(val_loss)
    # save the best model
    # if val_acc > best_accuracy:
    #     torch.save(model.state_dict(), os.path.join(data_dir,"output","MLTC_model_state.bin"))
    #     best_accuracy = val_acc

"""
# Evaluate the model using the test data
test_acc, test_loss = eval_model(test_data_loader, model, optimizer)



def get_predictions(model, data_loader):
    
    # Outputs:
    #   predictions - 
    
    model = model.eval()
    
    titles = []
    predictions = []
    prediction_probs = []
    target_values = []

    with torch.no_grad():
      for data in data_loader:
        title = data["title"]
        ids = data["input_ids"].to(device, dtype = torch.long)
        mask = data["attention_mask"].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data["targets"].to(device, dtype = torch.float)
        
        outputs = model(ids, mask, token_type_ids)
        # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
        outputs = torch.sigmoid(outputs).detach().cpu()
        # thresholding at 0.5
        preds = outputs.round()
        targets = targets.detach().cpu()

        titles.extend(title)
        predictions.extend(preds)
        prediction_probs.extend(outputs)
        target_values.extend(targets)
    
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    target_values = torch.stack(target_values)
    
    return titles, predictions, prediction_probs, target_values

"""


# example = test_df['File Contents'][0]

# # print(example)

# # model.compile(optimizer=optimizer)

# encodings = tokenizer.encode_plus(
#     example,
#     None,
#     add_special_tokens = True,
#     max_length = MAX_LEN,
#     truncation = True,
#     padding = "max_length", 
#     return_attention_mask = True, 
#     return_tensors = "pt"
# )

# model.eval()
# with torch.no_grad():
#     ids = encodings['input_ids'].to(device, dtype = torch.long)
#     mask = encodings['attention_mask'].to(device, dtype = torch.long)
#     token_type_ids = encodings['token_type_ids'].to(device, dtype = torch.long)
#     # targets = encodings['targets'].to(device, dtype = torch.float)
#     outputs = model(ids, mask, token_type_ids)
#     print(outputs)
#     outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
#     print(outputs)
