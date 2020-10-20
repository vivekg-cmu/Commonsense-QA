import sys

sys.path.append(".")

import torch
from transformers import DistilBertTokenizer, DistilBertModel


class DownstreamModel(torch.nn.Module):
    def __init__(self, vocab_size=31000):
        super(DownstreamModel, self).__init__()
        self.vocab_size = vocab_size
        self.distil = DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                      return_dict=True)
        self.cls_layer = torch.nn.Linear(768 * 3, 3)
        self.cls_loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, ans_a, ans_b, ans_c, labels):
        ans_a_cls = self.distil(ans_a).last_hidden_state[:, 0, :].squeeze(1)
        ans_b_cls = self.distil(ans_b).last_hidden_state[:, 0, :].squeeze(1)
        ans_c_cls = self.distil(ans_c).last_hidden_state[:, 0, :].squeeze(1)
        linear_input = torch.cat([ans_a_cls, ans_b_cls, ans_c_cls], dim=-1)
        answer_logits = self.cls_layer(linear_input)
        qa_loss = self.qa_loss(answer_logits=answer_logits, labels=labels)
        answer_preds = torch.argmax(answer_logits, dim=-1)
        return answer_preds, qa_loss

    def qa_loss(self, answer_logits, labels):
        return self.nsp_loss_func(input=answer_logits.view(-1, 2),
                                  target=labels.view(-1))


