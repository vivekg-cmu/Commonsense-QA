import sys
sys.path.append(".")
import pandas as pd
import json

class SocialIqaParser:

    def __init__(self):
        # File paths
        self.train_json_path = "Downstream/Data/train.jsonl"
        self.dev_json_path = "Downstream/Data/dev.jsonl"
        self.train_labels_path = "Downstream/Data/train-labels.lst"
        self.dev_labels_path = "Downstream/Data/dev-labels.lst"

    def read_json(self, file_path):
        with open(file_path, 'r') as fh:
            result = [json.loads(x) for x in fh.readlines()]
        return result

    def read_labels(self, file_path):
        with open(file_path, 'r') as fh:
            result = [int(x) for x in fh.readlines()]
        return result

    def convert_json_to_pandas(self, json_data):

        json_dict = {"ans_a": [],
                     "ans_b": [],
                     "ans_c": [],
                     "context": [],
                     "question": []
                     }
        for json_line in json_data:
            json_dict["ans_a"].append(json_line['answerA'])
            json_dict["ans_b"].append(json_line['answerB'])
            json_dict["ans_c"].append(json_line['answerC'])
            json_dict["context"].append(json_line['context'])
            json_dict["question"].append(json_line['question'])

        return pd.DataFrame.from_dict(json_dict)


    def merge_x_and_labels(self, pandas_x, labels):
        pandas_x['labels'] = labels
        return pandas_x

    def setup(self):
        json_data_train = self.read_json(self.train_json_path)
        json_data_dev = self.read_json(self.dev_json_path)
        train_y = self.read_labels(self.train_labels_path)
        dev_y = self.read_labels(self.dev_labels_path)
        train_pandas_x = self.convert_json_to_pandas(json_data_train)
        dev_pandas_x = self.convert_json_to_pandas(json_data_dev)
        train_pandas = self.merge_x_and_labels(train_pandas_x, train_y)
        dev_pandas = self.merge_x_and_labels(dev_pandas_x, dev_y)
        pd.DataFrame.to_csv(train_pandas, 'Downstream/Data/train_pandas.csv')
        pd.DataFrame.to_csv(dev_pandas, 'Downstream/Data/dev_pandas.csv')



