import sys

sys.path.append(".")

import torch
from transformers import DistilBertTokenizer, DistilBertModel


class PretrainingModel(torch.nn.Module):
    def __init__(self, vocab_size=31000):
        super(PretrainingModel, self).__init__()

        self.vocab_size = vocab_size
        # self.embedding = torch.nn.Embedding(vocab_size, 100)
        # self.rnn = torch.nn.LSTM(100, 100, batch_first=True)
        self.distil = DistilBertModel.from_pretrained('distilbert-base-uncased',
                                                      return_dict=True)
        self.mlm = torch.nn.Linear(100, 30000)
        self.nsp = torch.nn.Linear(100, 2)
        self.mlm_loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.nsp_loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, src, masked_lm_ids, masked_lm_positions, nsp_labels):
        # embedded = self.embedding(src)
        outputs = self.distil(src)
        # print(outputs)
        # outputs, _ = self.rnn(embedded)
        logits = self.mlm(outputs)
        masked_outputs = torch.stack([torch.index_select(logits[i], dim=0, index=masked_lm_positions[i])
                       for i in range(logits.shape[0])])
        next_sentence_outputs = self.nsp(outputs[:, 0, :])

        mlm_loss = self.mlm_loss(masked_outputs=masked_outputs,
                                 masked_lm_ids=masked_lm_ids)

        nsp_loss = self.nsp_loss(next_sentence_outputs, nsp_labels)
        return masked_outputs, next_sentence_outputs, mlm_loss, nsp_loss

    def mlm_loss(self, masked_outputs, masked_lm_ids):
        a = masked_outputs.view(-1, 30000)
        b = masked_lm_ids.view(-1)
        return self.mlm_loss_func(input=a, target=b)

    def nsp_loss(self, next_sentence_outputs, nsp_labels):
        return self.nsp_loss_func(input=next_sentence_outputs.view(-1, 2),
                                  target=nsp_labels.view(-1))


