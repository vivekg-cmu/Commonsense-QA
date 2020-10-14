import sys

sys.path.append(".")

import torch
from BERT_retraining.DataSetup.Preprocessors import Preprocessors
from BERT_retraining.Models.DistilBERT import PretrainingModel
from BERT_retraining import constants as con
from torch import optim
from tqdm import tqdm


class PretrainingTrainer:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.optimizer = None

    def setup_preprocessed_data(self):
        self.preprocessor = Preprocessors()
        self.preprocessor.get_loaders(load_flag=False)

    def setup_model(self):
        # Create multilingual vocabulary
        self.model = PretrainingModel()

        if con.CUDA:
            self.model = self.model.cuda()

    def setup_scheduler_optimizer(self):
        lr_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr_rate, weight_decay=0)

    def train_model(self):
        train_loader = self.preprocessor.train_loaders
        batch_size = 8

        self.model.train()
        train_loss = 0
        batch_correct = 0
        total_correct = 0
        index = 0
        for input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, \
            next_sentence_labels in train_loader:
            masked_outputs, next_sentence_outputs, mlm_loss, nsp_loss = self.model(input_ids, masked_lm_ids,
                                                                                   masked_lm_positions,
                                                                                   next_sentence_labels)
            batch_loss = mlm_loss + nsp_loss


            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            batch_correct += self.evaluate(masked_outputs=masked_outputs, masked_lm_ids=masked_lm_ids)
            total_correct += (8 * 20)
            index += 1

            if index % 50 == 0:
                print("total_loss", batch_loss)
                print(batch_correct /  total_correct)

                print('------------------------')

    def evaluate(self, masked_outputs, masked_lm_ids):
        masked_output_predictions = torch.argmax(masked_outputs, dim=-1).view(-1)
        masked_lm_ids = masked_lm_ids.view(-1)
        correct = torch.sum(torch.eq(masked_output_predictions, masked_lm_ids))


        return correct.numpy()



    def run_pretraining(self):
        self.setup_preprocessed_data()
        self.setup_model()
        self.setup_scheduler_optimizer()
        self.train_model()


if __name__ == '__main__':
    trainer = PretrainingTrainer()
    trainer.run_pretraining()

