import sys

sys.path.append(".")

import constants as con
from DataSetup.dataset import BERTDataset, collate_mlm
from torch.utils.data import DataLoader
from DataSetup.WordVocab import WordVocab


class Dataloaders:

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

    def get_datasets(self):
        vocab = WordVocab.load_vocab(self.vocab_path)
        print("Loading Train Dataset", self.paths[con.TRAIN_PATH])
        self.train_dataset = BERTDataset(self.paths[con.TRAIN_PATH],
                                         vocab,
                                         corpus_lines=100)

        print("Loading Valid Dataset", self)
        self.valid_dataset = BERTDataset(self.paths[con.VALID_PATH], vocab)

    def get_dataloaders(self):
        print("Creating Dataloader")
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                            collate_fn=lambda batch: collate_mlm(batch), shuffle=False)

        self.valid_data_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                            collate_fn=lambda batch: collate_mlm(batch), shuffle=False)

    def setup(self):
        self.get_datasets()
        self.get_dataloaders()


if __name__ == '__main__':
    dataloader = Dataloaders()
    dataloader.setup()
    print(dataloader.train_data_loader)
    for k in dataloader.train_data_loader:
        print(k)