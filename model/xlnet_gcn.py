import torch.nn as nn
import torch.nn.functional as F

from .xlnet import XLNET
from .gcn import GCN

class XLNetGCN(nn.Module):
    def __init__(self, num_classes = 23, m = 0.7, gcn_layers = 2, gcn_hidden = 200, gcn_dropout = 0.5, gcn_norm = 'none'):
        super(XLNetGCN, self).__init__()
        self.m = m
        self.num_classes = num_classes
        self.xlnet = XLNET()
        self.tokenizer = self.xlnet.get_tokenizer()
        self.gcn = GCN(input_features = 768, n_hidden = gcn_hidden, n_classes = num_classes, n_layers = gcn_layers - 1, activation = F.elu, dropout = gcn_dropout, normalization = gcn_norm)
        self.classifier = nn.Linear(768, num_classes)
        

    def forward(self, input_ids, attn_mask, token_type_ids, g, h):
        """
        :param input_ids: input ids for the XLNet model
        :param attn_mask: attention mask for the XLNet model
        :param token_type_ids: token type ids for the XLNet model
        :param g: graph structure in the form of a DGL graph
        :param h: node features for the graph
        """
        # Assuming idx is a list of indices for which we have the input data
        # and we want to use XLNet to update their features in the graph.

        # XLNet Processing
        # Process input data through XLNet
        xlnet_output = self.xlnet(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        cls_feats = xlnet_output[0][:, 0]  # Get the [CLS] token's embeddings for all instances

        # Update the features of the nodes in idx with XLNet embeddings
        # This step assumes h is the initial feature matrix for all nodes in the graph
        # and we're updating the features of nodes in idx with XLNet embeddings.
        # If h is not initially provided or needs to be constructed, you'll need to adjust this part.
        h['idx'] = cls_feats['idx']

        # GCN Processing
        # Apply GCN on the entire graph with updated features
        gcn_output = self.gcn(g, h)

        # Combine XLNet and GCN outputs for the final prediction
        # Here, we use a simple approach of averaging the outputs from both models as an example.
        # You might explore more sophisticated methods for combining these outputs depending on your task.
        combined_output = self.m * cls_feats['idx'] + (1 - self.m) * gcn_output['idx']

        # Pass the combined output through a classifier (if necessary)
        # Note: Depending on your setup, you might need to adjust the dimensions or process the combined_output further.
        predictions = self.classifier(combined_output)

        return predictions



    def get_tokenizer(self):
        return self.xlnet.get_tokenizer()

    def get_xlnet(self):
        return self.xlnet

    def get_gcn(self):
        return self.gcn