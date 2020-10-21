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
        # for param in self.distil.parameters():
        #   param.requires_grad = False

        self.cls_layer = torch.nn.Linear(768 * 3, 768)
        self.cls_layer2 = torch.nn.Linear(768, 256)
        self.cls_layer3 = torch.nn.Linear(256, 64)
        self.cls_layer4 = torch.nn.Linear(64, 32)
        self.cls_layer5 = torch.nn.Linear(32, 4)
        self.cls_loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, ans_a, ans_b, ans_c, labels):
        ans_a_cls = self.distil(ans_a).last_hidden_state[:, 0, :].squeeze(1)
        ans_b_cls = self.distil(ans_b).last_hidden_state[:, 0, :].squeeze(1)
        ans_c_cls = self.distil(ans_c).last_hidden_state[:, 0, :].squeeze(1)
        linear_input = torch.cat([ans_a_cls, ans_b_cls, ans_c_cls], dim=-1)
        answer_logits = self.cls_layer(linear_input)
        answer_logits = self.cls_layer2(answer_logits)
        answer_logits = self.cls_layer3(answer_logits)
        answer_logits = self.cls_layer4(answer_logits)
        answer_logits = self.cls_layer5(answer_logits)

        qa_loss = self.qa_loss_f(answer_logits=answer_logits, labels=labels)
        answer_preds = torch.argmax(answer_logits, dim=-1)
        return answer_preds, qa_loss

    def qa_loss_f(self, answer_logits, labels):
        return self.cls_loss_func(input=answer_logits,
                                  target=labels)


