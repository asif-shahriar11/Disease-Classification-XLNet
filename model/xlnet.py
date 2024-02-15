import torch.nn as nn
from transformers import XLNetModel, XLNetTokenizer

class XLNET(nn.Module):
    def __init__(self):
        super(XLNET, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 23)  # Assuming 23 classes for classification

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get the full output
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # The last hidden state is the first element of the output
        last_hidden_state = outputs[0]

        # Pooling operation if needed, for example, using the [CLS] token's embedding
        # Assuming [CLS] is the first token, similar to BERT. Adjust as needed.
        pooled_output = last_hidden_state[:, 0]

        output = self.drop(pooled_output)
        return self.out(output)
    
    def get_tokenizer(self):
        return XLNetTokenizer.from_pretrained('xlnet-base-cased')