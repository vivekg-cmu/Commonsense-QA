import torch
import numpy as np

class PretrainingModel(torch.nn.Module):
    def __init__(self, vocab_size=31000):
        super(PretrainingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, 100)
        self.rnn = torch.nn.LSTM(100, 100, batch_first=True)
        self.fc = torch.nn.Linear(100, 50)
        self.soft_layer = torch.nn.Softmax(dim=2)

    def forward(self, src, masked_lm_ids, masked_lm_positions):
        print("Source", src)
        embedded = self.embedding(src)
        outputs, _ = self.rnn(embedded)
        print("Outputs", outputs.shape)
        linear = self.fc(outputs)
        print("Linear", linear.shape)

        logits = self.soft_layer(linear)
        print("Post softmax", torch.sum(logits[0][0]))
        print(masked_lm_ids.shape)
        print(masked_lm_positions.shape)

        masked_outputs = torch.stack([torch.index_select(logits[i], dim=0, index=masked_lm_positions[i])
                       for i in range(8)])
