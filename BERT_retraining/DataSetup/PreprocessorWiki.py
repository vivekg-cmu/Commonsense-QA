import sys
sys.path.append(".")
from BERT_retraining.DataSetup.Pretraining import Pretraining
from BERT_retraining.DataSetup.Dataloaders import PretrainingDataset
from BERT_retraining import utils
from torch.utils import data
from BERT_retraining import constants as con

class PreprocessorsWiki:

    def __init__(self):

        self.train_loaders = None
        self.valid_loaders = None

    def run_pretraining(self, path="BERT_retraining/Data/clean_wikihow_", key="train"):
        pretraining = Pretraining(
            vocab_file="BERT_retraining/Data/bert-base-uncased-vocab.txt",
            do_lower_case=True,
            input_file=path + key + ".txt",
            random_seed=12345,
            max_seq_length=32,
            dupe_factor=4,
            max_predictions_per_seq=5,
            masked_lm_prob=0.15,
            output_file="Data/wikihow_output.txt",
            short_seq_prob=0.1)

        instances, tokenizer = pretraining.run_data_preprocessing()
        features = pretraining.write_instance_to_features(instances=instances, tokenizer=tokenizer,
                                                          max_seq_length=32,
                                                          max_predictions_per_seq=5)
        utils.save_dictionary(dictionary=features,
                              save_path="./BERT_retraining/Data/"
                                        + key + "_wiki.pkl")
        return features

    def get_features(self, load_flag=False):
        if load_flag:
            train_features = \
                utils.load_dictionary("BERT_retraining/Data/train_wiki.pkl")
            valid_features = \
                utils.load_dictionary("BERT_retraining/Data/valid_wiki.pkl")
        else:
            train_features, valid_features = self.run_features()

        return train_features, valid_features

    def run_features(self):
        train_features = self.run_pretraining(key='train')
        valid_features = self.run_pretraining(key='valid')
        utils.save_dictionary(dictionary=train_features,
                              save_path="BERT_retraining/Data/train_wiki.pkl")
        utils.save_dictionary(dictionary=valid_features,
                              save_path="BERT_retraining/Data/valid_wiki.pkl")
        return train_features, valid_features

    def get_loaders(self, load_flag=False):
        train_features, valid_features = self.get_features(load_flag=load_flag)
        train_dataset = PretrainingDataset(input_dict=train_features)
        valid_dataset = PretrainingDataset(input_dict=valid_features)

        loader_args = dict(shuffle=True, batch_size=con.BATCH_SIZE)

        self.train_loaders = data.DataLoader(train_dataset, **loader_args)
        self.valid_loaders = data.DataLoader(valid_dataset, **loader_args)


if __name__ == '__main__':
    PreprocessorsWiki().get_loaders(load_flag=False)
