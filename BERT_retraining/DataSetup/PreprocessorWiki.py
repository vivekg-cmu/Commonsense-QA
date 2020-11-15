import sys
sys.path.append(".")
from BERT_retraining.DataSetup.Pretraining import Pretraining
from BERT_retraining.DataSetup.Dataloaders import PretrainingDataset
from BERT_retraining import utils
from torch.utils import data
from BERT_retraining import constants as con
import glob

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
            max_seq_length=128,
            dupe_factor=1,
            max_predictions_per_seq=20,
            masked_lm_prob=0.15,
            output_file="not_used",
            short_seq_prob=0.1)

        instances, tokenizer = pretraining.run_data_preprocessing()
        features = pretraining.write_instance_to_features(instances=instances, tokenizer=tokenizer,
                                                          max_seq_length=128,
                                                          max_predictions_per_seq=20)
        utils.save_dictionary(dictionary=features,
                              save_path="./BERT_retraining/Data/"
                                        + key + "_wiki.pkl")
        return features

    def run_batch_features(self):
        for batch_number in range(200):
            self.run_pretraining(key=str(batch_number))

    def load_pickle_files(self, key=0):
        train_features = {
            "input_ids": [],
            "input_mask": [],
            "segment_ids": [],
            "masked_lm_positions": [],
            "masked_lm_ids": [],
            "masked_lm_weights": [],
            "next_sentence_labels": []
        }

        if key == 0:
            start = 0
            end = 100
        else:
            start = 100
            end = 199
        for index in range(start, end):
            features = utils.load_dictionary(f"BERT_retraining/Data/{index}_wiki.pkl")
            for key in train_features:
                train_features[key].extend(features[key])

        valid_features = utils.load_dictionary("BERT_retraining/Data/199_wiki.pkl")
        return train_features, valid_features

    def get_loaders(self, key=0):
        train_features, valid_features = self.load_pickle_files(key=0)
        train_dataset = PretrainingDataset(input_dict=train_features)
        valid_dataset = PretrainingDataset(input_dict=valid_features)
        loader_args = dict(shuffle=True, batch_size=con.BATCH_SIZE)
        self.train_loaders = data.DataLoader(train_dataset, **loader_args)
        self.valid_loaders = data.DataLoader(valid_dataset, **loader_args)


if __name__ == '__main__':
    PreprocessorsWiki().run_batch_features()
