import sys

sys.path.append(".")
import pandas as pd
from transformers import DistilBertTokenizer
from tqdm import tqdm
from Downstream.utils import save_dictionary


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
                "ans_c": [],
                "label": []
            },
            "valid": {
                "ans_a": [],
                "ans_b": [],
                "ans_c": [],
                "label": []
            },
            "test": {
                "ans_a": [],
                "ans_b": [],
                "ans_c": [],
                "label": []
            }
        }

    def setup_tokenizer(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',
                                                             max_length=256, padding="max_length",
                                                             truncation=True, is_split_into_words=False)
        print(self.tokenizer)

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.valid_data = pd.read_csv(self.valid_path)
        self.test_data = pd.read_csv(self.test_path)

    def tokenize_data(self, data, key):

        questions = data['question']
        labels = data['labels']
        contexts = data['context']

        for i in tqdm(range(len(data))):
            for ans in ["ans_a", "ans_b", "ans_c"]:
                answer = self.tokenizer(text=data[ans][i])['input_ids']
                context = self.tokenizer(text=contexts[i])['input_ids'][1:]
                question = self.tokenizer(text=questions[i])['input_ids'][1:]
                input_line = answer + context + question
                self.input_dict[key][ans].append(input_line + [0 for _ in range(512 - len(input_line))])
            self.input_dict[key]["label"].append(labels[i])

    def get_loaders(self, load_flag=False):
        train_features, valid_features = self.get_features(load_flag=load_flag)
        train_dataset = PretrainingDataset(input_dict=train_features)
        valid_dataset = PretrainingDataset(input_dict=valid_features)

        loader_args = dict(shuffle=True, batch_size=con.BATCH_SIZE)

        self.train_loaders = data.DataLoader(train_dataset, **loader_args)
        self.valid_loaders = data.DataLoader(valid_dataset, **loader_args)

    def setup(self):
        self.load_data()
        self.setup_tokenizer()
        self.tokenize_data(data=self.train_data, key="train")
        self.tokenize_data(data=self.valid_data, key="valid")
        self.tokenize_data(data=self.test_data, key="test")
        save_dictionary(dictionary=self.input_dict,
                        save_path="Downstream/Data/input_dict.pkl")


if __name__ == '__main__':
    p = Preprocessor()
    p.setup()
