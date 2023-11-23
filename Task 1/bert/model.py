import torch
from transformers import BertForTokenClassification


class BertModel(torch.nn.Module):
    def __init__(self, classes_num: int=3, weight_path: str=None):
        super(BertModel, self).__init__()
        if weight_path is None:
            self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=classes_num)
        else:
            self.bert = BertForTokenClassification.from_pretrained(weight_path, num_labels=classes_num)


    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output
