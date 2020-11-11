import sys

sys.path.append(".")
from CommonSense.utils import load_dictionary, save_dictionary
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils import data
from CommonSense.Preprocess.Dataloaders import DownstreamDataset
from CommonSense import constants as con

class Preprocessor:
    options = ['A', 'B', 'C', 'D', 'E']
    max_seq_len = 128

    def __init__(self):
        self.input_dict = load_dictionary("CommonSense/Data/input_dict.pkl")
        self.tokenized_dict = {
            "train":{
                opt: [] for opt in Preprocessor.options + ['question', 'label', 'concept']
                                   + [x + "_att" for x in Preprocessor.options]

            },
            "dev": {
                opt: [] for opt in Preprocessor.options + ['question', 'label', 'concept']
                                   + [x + "_att" for x in Preprocessor.options]
            }
        }
        self.tokenzier = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_data(self, data_dict, key="train"):
        for i in tqdm(range(len(data_dict['question']))):
            question = self.tokenzier(data_dict['question'][i])['input_ids']
            context = self.tokenzier(data_dict['concept'][i])['input_ids'][1:]

            for option in Preprocessor.options:
                answer = self.tokenzier(data_dict[option][i])['input_ids'][1:]
                input_line = question + context + answer

                self.tokenized_dict[key][option].append(input_line + [0 for _ in range(Preprocessor.max_seq_len - len(input_line))])
                self.tokenized_dict[key][option + '_att'].append([1 for _ in range(len(input_line))] + [0 for _ in range(Preprocessor.max_seq_len - len(input_line))])
            self.tokenized_dict[key]["label"].append(data_dict['y'][i])

    def get_loaders(self, load_from_pkl=False):
        if load_from_pkl:
            try:
                self.tokenized_dict = load_dictionary("CommonSense/Data/tokenized_dict.pkl")
            except Exception as e:
                print(e)

        train_dataset = DownstreamDataset(input_dict=self.tokenized_dict["train"])
        valid_dataset = DownstreamDataset(input_dict=self.tokenized_dict["dev"])
        loader_args = dict(shuffle=True, batch_size=con.BATCH_SIZE)
        self.train_loaders = data.DataLoader(train_dataset, **loader_args)
        self.valid_loaders = data.DataLoader(valid_dataset, **loader_args)

    def setup(self):
        self.tokenize_data(data_dict=self.input_dict['train'], key='train')
        self.tokenize_data(data_dict=self.input_dict['dev'], key='dev')
        save_dictionary(dictionary=self.tokenized_dict,
                        save_path="CommonSense/Data/tokenized_dict.pkl")
        self.get_loaders(load_from_pkl=True)


if __name__ == '__main__':
    p = Preprocessor()
    p.setup()










