import sys
sys.path.append(".")
import pandas as pd
from transformers import DistilBertTokenizer


class Preprocessor:

    def __init__(self):
        self.train_path = "Downstream/Data/dev_data.csv"
        self.valid_path = "Downstream/Data/dev_data.csv"
        self.test_path = "Downstream/Data/dev_data.csv"

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.tokenizer = None

        self.input_dict = {
            "train": {
                "ans_a": [],
                "ans_b": [],
                "ans_c": []
            },
            "valid": {
                "ans_a": [],
                "ans_b": [],
                "ans_c": []
            },
            "test": {
                "ans_a": [],
                "ans_b": [],
                "ans_c": []
            }
        }

    def setup_tokenizer(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',
                                                             max_length=256, padding="max_length",
                                                             truncation=True)
        print(self.tokenizer)

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.valid_data = pd.read_csv(self.valid_path)
        self.test_data = pd.read_csv(self.test_path)

    def tokenize_data(self, data, key):

        question = data['question']
        labels = data['labels']
        context = data['context']
        ans_a = data["ans_a"]
        ans_b = data["ans_b"]
        ans_c = data["ans_c"]

        for i in range(len(data)):
            self.input_dict[key]["ans_a"].append([])
            print(self.tokenizer(ans_a[i], return_tensors="pt"))

            break

    def setup(self):
        self.load_data()
        self.setup_tokenizer()
        self.tokenize_data(data=self.train_data, key="train")


if __name__ == '__main__':
    p = Preprocessor()
    p.setup()
