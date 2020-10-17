import sys
sys.path.append(".")
import pandas as pd

class ConceptNet:

    def __init__(self, base_path="BERT_retraining/Concepnet/"):
        self.base_path = base_path
        self.conceptnet_data = []

    def clean_data(self, text):
        text = text.rstrip().strip()
        text = " ".join(text.split('\t')[:-1])
        text = text.lower()
        return text

    def write_data(self):
        with open(self.base_path + "concept_train.txt", 'w') as fh:
            fh.writelines([x + '\n' for x in self.conceptnet_data])

    def load_concept_data(self):
        extensions = ["train100k.txt", "train300k.txt", "train600k.txt"]
        self.conceptnet_data = []
        for extension in extensions:
            with open(self.base_path + extension, 'r') as fh:
                self.conceptnet_data.extend(fh.readlines())
        self.conceptnet_data = [self.clean_data(text) for text in self.conceptnet_data]

    def setup(self):
        self.load_concept_data()
        self.write_data()


if __name__ == '__main__':
    c = ConceptNet()
    c.setup()










