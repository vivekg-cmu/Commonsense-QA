from BERT_retraining.create_pretraining_data import create_training_instances, \
    write_instance_to_example_files
from BERT_retraining import tokenization
import tensorflow as tf
import random
from BERT_retraining import utils


class Pretraining:
    """
    This method is use to preprocess the data and convert it to the masked language model and
    next sentence prediction task
    """
    def __init__(self, vocab_file, do_lower_case, input_file, random_seed, max_seq_length,
                 dupe_factor, short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                 output_file):
        """
        Args:
            vocab_file (str): Path where the vocab is present
            do_lower_case (bool): This flag is used to select lowercase of a model
            input_file (str): File to input_files string which is comma separated
            random_seed (int): seed to initiaize masking
            max_seq_length (int): maximum sequence length
            dupe_factor (int): duplication for each sentence
            short_seq_prob (float): sequence probabilty for each
            masked_lm_prob (float): masked language probability
            max_predictions_per_seq (int): maximum predictions
            output_file (str): the output file
        """
        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case
        self.input_file = input_file
        self.random_seed = random_seed
        self.max_seq_length = max_seq_length
        self.dupe_factor = dupe_factor
        self.short_seq_prob = short_seq_prob
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.output_file = output_file

    def run_data_preprocessing(self):
        """
        This method is used to run preprocessing
        Returns:
            instances (Instances): the instances consiting of the complete data
        """
        tf.logging.set_verbosity(tf.logging.INFO)

        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

        input_files = []
        for input_pattern in self.input_file.split(","):
            input_files.extend(tf.gfile.Glob(input_pattern))

        tf.logging.info("*** Reading from input files ***")
        for input_file in input_files:
            tf.logging.info("  %s", input_file)

        rng = random.Random(self.random_seed)
        instances = create_training_instances(
            input_files, tokenizer, self.max_seq_length, self.dupe_factor,
            self.short_seq_prob, self.masked_lm_prob, self.max_predictions_per_seq,
            rng)

        return instances, tokenizer

    def write_output(self, instances, tokenizer):
        output_files = self.output_file.split(",")
        tf.logging.info("*** Writing to output files ***")
        for output_file in output_files:
            tf.logging.info("  %s", output_file)

        write_instance_to_example_files(instances, tokenizer, self.max_seq_length,
                                        self.max_predictions_per_seq, output_files)

    @staticmethod
    def write_instance_to_features(instances, tokenizer, max_seq_length,
                                        max_predictions_per_seq):
        """Create TF example files from `TrainingInstance`s."""
        features = {
            "input_ids" :[],
        "input_mask" : [],
        "segment_ids": [],
        "masked_lm_positions": [],
        "masked_lm_ids": [],
        "masked_lm_weights": [],
        "next_sentence_labels": []
        }

        for (inst_index, instance) in enumerate(instances):
            input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = list(instance.segment_ids)
            assert len(input_ids) <= max_seq_length

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            masked_lm_positions = list(instance.masked_lm_positions)
            masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
            masked_lm_weights = [1.0] * len(masked_lm_ids)

            while len(masked_lm_positions) < max_predictions_per_seq:
                masked_lm_positions.append(0)
                masked_lm_ids.append(0)
                masked_lm_weights.append(0.0)

            next_sentence_label = 1 if instance.is_random_next else 0

            features["input_ids"].append(input_ids)
            features["input_mask"].append(input_mask)
            features["segment_ids"].append(segment_ids)
            features["masked_lm_positions"].append(masked_lm_positions)
            features["masked_lm_ids"].append(masked_lm_ids)
            features["masked_lm_weights"].append(masked_lm_weights)
            features["next_sentence_labels"].append([next_sentence_label])


        return features


if __name__ == '__main__':
    MAX_SEQ_LEN = 128
    MAX_PREDS = 20

    preprocess = Pretraining(
        vocab_file="/home/pratik/Desktop/new_github/Commonsense-QA/BERT_retraining/Data/bert-base-uncased-vocab.txt",
        do_lower_case=True,
        input_file="/home/pratik/Desktop/new_github/Commonsense-QA/BERT_retraining/Data/train.txt",
        random_seed=12345,
        max_seq_length=128,
        dupe_factor=4,
        max_predictions_per_seq=20,
        masked_lm_prob=0.15,
        output_file="Data/temp.txt",
        short_seq_prob=0.1)

    instances, tokenizer = preprocess.run_data_preprocessing()
    features = preprocess.write_instance_to_features(instances=instances, tokenizer=tokenizer,
                                          max_seq_length=MAX_SEQ_LEN,
                                          max_predictions_per_seq=MAX_PREDS)

    utils.save_dictionary(dictionary=features,
        save_path="/home/pratik/Desktop/new_github/Commonsense-QA/BERT_retraining/Data/features.pkl")
    loaded_features = utils.load_dictionary("/home/pratik/Desktop/new_github/Commonsense-QA/BERT_retraining/Data/features.pkl")
    print(loaded_features['input_ids'][0])