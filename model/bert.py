import torch
from transformers import BertTokenizer, BertModel

# bert model
class BERTClass(torch.nn.Module):
    def __init__(self, dropout=0.25, num_classes=23):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output
    
    def get_tokenizer(self):
        return BertTokenizer.from_pretrained('bert-base-uncased')

