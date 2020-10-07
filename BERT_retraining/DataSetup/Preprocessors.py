import sys
sys.path.append(".")
from BERT_retraining import constants as con
from torch.utils.data import DataLoader


class Preprocessors:

    def __init__(self):
        self.paths = {
            con.TRAIN_PATH: "Data/train.txt",
            con.VALID_PATH: "Data/train.txt"
        }
        self.vocab_path = "Data/bert-base-uncased-vocab.txt"

        self.train_dataset = None
        self.valid_dataset = None
        self.batch_size = 8

        self.train_data_loader = None
        self.valid_data_loader = None


if __name__ == '__main__':
    dataloader = Dataloaders()
    dataloader.setup()
    print(dataloader.train_data_loader)
    for k in dataloader.train_data_loader:
        print(k)