import torch
from BERT_retraining.DataSetup.Preprocessors import Preprocessors
from BERT_retraining.Models.Pretraining import PretrainingModel
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
        self.preprocessor.get_loaders(load_flag=True)

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
        for input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, \
            next_sentence_labels in train_loader:

            self.model(input_ids, masked_lm_ids, masked_lm_positions)
            break
            # self.optimizer.zero_grad()
            # batch_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            # self.optimizer.step()

    def run_pretraining(self):
        self.setup_preprocessed_data()
        self.setup_model()
        self.setup_scheduler_optimizer()
        self.train_model()


if __name__ == '__main__':
    trainer = PretrainingTrainer()
    trainer.run_pretraining()

