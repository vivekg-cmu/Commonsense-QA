import sys

sys.path.append(".")
import pandas as pd
import re
from tqdm import tqdm


class ParsePiqa:
    def __init__(self, source_path="./BERT_retraining/Data/wikihowSep.csv",
                 dest_path="./BERT_retraining/Data/clean_wikihow_"):
        self.source_path = source_path
        self.output_writer_train = open(dest_path + "train.txt", "w")
        self.output_writer_valid = open(dest_path + "valid.txt", "w")
        self.data = None

    def load_wikihow(self):
        self.data = pd.read_csv(self.source_path)
        self.data.dropna(inplace=True)

    def parse_wikihow(self, text, output_writer):
        try:
            text = "".join([x.rstrip().strip() for x in text.split("\n") if x])

            text_list = [x.strip() + '\n' for x in text.split('.') if len(re.findall("\w+", x)) > 5]
            if text_list:
                text_list[-1] += '\n'
                output_writer.writelines(text_list)
        except Exception as e:
            print(e)
            return False

        return True

    def setup(self):
        self.load_wikihow()
        error_count = 0
        index = 0
        for text in tqdm(self.data['text']):

            if index < int(len(self.data) * 0.95):
                error_status = self.parse_wikihow(text, self.output_writer_train)
            else:
                error_status = self.parse_wikihow(text, self.output_writer_valid)

            if not error_status:
                error_count += 1

            index += 1
        print(f"Parsing completed with {error_count} errors")


if __name__ == '__main__':
    parse_piqa = ParsePiqa()
    parse_piqa.setup()

