import sys

sys.path.append(".")
import ast


class ParseCommonSense:
    options = ['A', 'B', 'C', 'D', 'E']

    def __init__(self):
        self.train_json_path = "CommonSense/Data/train_rand_split.jsonl"
        self.dev_json_path = "CommonSense/Data/dev_rand_split.jsonl"

        self.train_data = {opt: [] for opt in ParseCommonSense.options}
        self.train_data['question'] = []
        self.train_data['concept'] = []
        self.train_data['y'] = []

        self.dev_data = {opt: [] for opt in ParseCommonSense.options}
        self.dev_data['question'] = []
        self.dev_data['concept'] = []
        self.dev_data['y'] = []

    def load_json_data(self, data):
        with open(self.train_json_path, 'r') as fh:
            json_data = fh.readlines()
        for json_line in json_data:
            json_line = ast.literal_eval(json_line)
            data['y'].append(ParseCommonSense.options.index(json_line['answerKey']))
            question = json_line['question']
            data['question'].append(question['stem'])
            data['concept'].append(question['question_concept'])
            choices = json_line['question']['choices']

            for choice in choices:
                label = choice['label']
                data[label].append(choice['text'])

    def setup(self):
        self.load_json_data(self.train_data)
        self.load_json_data(self.dev_data)


pcs = ParseCommonSense()
pcs.load_json_data()

