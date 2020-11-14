import sys

sys.path.append(".")
import pandas as pd
import re
from tqdm import tqdm


class ParsePiqa:
    def __init__(self, source_path="./BERT_retraining/Data/wikihowSep.csv",
                 dest_path="./BERT_retraining/Data/clean_wikihow_"):
        self.source_path = source_path
        self.dest_path = dest_path
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
        index = 0
        batch_number = 0

        for text in tqdm(self.data['text']):
            if index % 200 == 0:
                output_writer = open(self.dest_path + str(batch_number), 'w')
                batch_number += 1
            self.parse_wikihow(text, output_writer)
            index += 1

if __name__ == '__main__':
    parse_piqa = ParsePiqa()
    parse_piqa.setup()

