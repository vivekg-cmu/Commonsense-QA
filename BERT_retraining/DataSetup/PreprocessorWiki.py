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

if __name__ == '__main__':
    PreprocessorsWiki().run_batch_features()
