from BERT_retraining.create_pretraining_data import create_training_instances, \
    write_instance_to_example_files
from BERT_retraining import tokenization
import tensorflow as tf
import random


class Preprocess:
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
            do_lower_case:
            input_file:
            random_seed:
            max_seq_length:
            dupe_factor:
            short_seq_prob:
            masked_lm_prob:
            max_predictions_per_seq:
            output_file:
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

    def run_pretraining(self):
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

        return instances

    def write_output(self, instances, tokenizer):
        output_files = self.output_file.split(",")
        tf.logging.info("*** Writing to output files ***")
        for output_file in output_files:
            tf.logging.info("  %s", output_file)

        write_instance_to_example_files(instances, tokenizer, self.max_seq_length,
                                        self.max_predictions_per_seq, output_files)


