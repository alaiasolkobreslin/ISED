from transformers import BertForTokenClassification

import torch


class BertModel(torch.nn.Module):

    def __init__(self, unique_labels):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(
            'bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask):

        output = self.bert(input_ids=input_id,
                           attention_mask=mask, return_dict=False)

        return output
