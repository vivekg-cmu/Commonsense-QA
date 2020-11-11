import sys

sys.path.append(".")
import torch
from CommonSense.Preprocess.Preprocessor import Preprocessor
from CommonSense.Models.DistilBERT import DownstreamModel
from SocialIQA import constants as con
from torch import optim


class CommonSenseTrainer:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.optimizer = None

    def setup_preprocessed_data(self):
        self.preprocessor = Preprocessor()
        self.preprocessor.get_loaders(load_from_pkl=True)

    def setup_model(self):
        # Create multilingual vocabulary
        self.model = DownstreamModel()

        if con.CUDA:
            self.model = self.model.cuda()

    def setup_scheduler_optimizer(self):
        lr_rate = 1e-5
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr_rate, weight_decay=0)

    def train_model(self):
        train_loader = self.preprocessor.train_loaders

        self.model.train()
        total_loss = 0
        batch_correct = 0
        total_correct = 0
        index = 0
        for ans_a, ans_b, ans_c, ans_d, ans_e, ans_a_att, ans_b_att, ans_c_att, \
            ans_d_att, ans_e_att, label in train_loader:
            self.optimizer.zero_grad()
            if con.CUDA:
                ans_a = ans_a.cuda()
                ans_b = ans_b.cuda()
                ans_c = ans_c.cuda()
                ans_d = ans_d.cuda()
                ans_e = ans_e.cuda()

                ans_a_att = ans_a_att.cuda()
                ans_b_att = ans_b_att.cuda()
                ans_c_att = ans_c_att.cuda()
                ans_d_att = ans_d_att.cuda()
                ans_e_att = ans_e_att.cuda()


                label = label.cuda()

            answer_preds, qa_loss = self.model(ans_a, ans_b, ans_c, ans_d, ans_e,
                                               ans_a_att, ans_b_att, ans_c_att,
                                               ans_d_att, ans_e_att,
                                               label)
            total_loss += qa_loss.cpu().detach().numpy()
            qa_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            batch_correct += self.evaluate(answer_preds=answer_preds,
                                           labels=label)
            total_correct += con.BATCH_SIZE
            index += 1

            # if index % 100 == 0:
            #     print("Running train loss", total_loss / index)
            #     print("Running train acc", batch_correct / total_correct)

        print("Train loss:", total_loss / index)
        print("Train acc:", batch_correct / total_correct)

    def valid_model(self):
        valid_loader = self.preprocessor.valid_loaders

        self.model.eval()
        total_loss = 0
        batch_correct = 0
        total_correct = 0
        index = 0
        for ans_a, ans_b, ans_c, ans_d, ans_e, ans_a_att, ans_b_att, ans_c_att, \
            ans_d_att, ans_e_att, label in valid_loader:
            self.optimizer.zero_grad()
            if con.CUDA:
                ans_a = ans_a.cuda()
                ans_b = ans_b.cuda()
                ans_c = ans_c.cuda()
                ans_d = ans_d.cuda()
                ans_e = ans_e.cuda()

                ans_a_att = ans_a_att.cuda()
                ans_b_att = ans_b_att.cuda()
                ans_c_att = ans_c_att.cuda()
                ans_d_att = ans_d_att.cuda()
                ans_e_att = ans_e_att.cuda()


                label = label.cuda()

            answer_preds, _ = self.model(ans_a, ans_b, ans_c, ans_d, ans_e,
                                               ans_a_att, ans_b_att, ans_c_att,
                                               ans_d_att, ans_e_att,
                                               label)

            batch_correct += self.evaluate(answer_preds=answer_preds,
                                           labels=label)
            total_correct += con.BATCH_SIZE
            index += 1

            # if index % 100 == 0:
            #     print("Running train loss", total_loss / index)
            #     print("Running train acc", batch_correct / total_correct)

        print("Valid loss:", total_loss / index)
        print("Valid acc:", batch_correct / total_correct)
        print('-----------------------------------------')

    def evaluate(self, answer_preds, labels):
        labels = labels.view(-1)
        correct = torch.sum(torch.eq(answer_preds, labels))
        return correct.cpu().detach().numpy()

    def run_pretraining(self):
        self.setup_preprocessed_data()
        self.setup_model()
        self.setup_scheduler_optimizer()
        for epoch in range(10):
            print("Epoch:", epoch)
            self.train_model()
            self.valid_model()


if __name__ == '__main__':
    trainer = CommonSenseTrainer()
    trainer.run_pretraining()
