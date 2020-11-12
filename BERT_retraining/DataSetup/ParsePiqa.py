import sys

sys.path.append(".")
import pandas as pd
import re
from tqdm import tqdm


class ParsePiqa:
    def __init__(self, source_path="./BERT_retraining/Data/wikihowSep.csv",
                 dest_path="./BERT_retraining/Data/clean_wikihow.txt"):
        self.source_path = source_path
        self.output_writer = open(dest_path, "w")
        self.data = None

    def load_wikihow(self):
        self.data = pd.read_csv(self.source_path)
        self.data.dropna(inplace=True)

    def parse_wikihow(self, text):
        try:
            text = "".join([x.rstrip().strip() for x in text.split("\n") if x])

            text_list = [x.strip() + '\n' for x in text.split('.') if len(re.findall("\w+", x)) > 5]
            if text_list:
                text_list[-1] += '\n'
                self.output_writer.writelines(text_list)
        except Exception as e:
            print(e)
            return False

        return True

    def setup(self):
        self.load_wikihow()
        error_count = 0
        for text in tqdm(self.data['text']):
            if not self.parse_wikihow(text):
                error_count += 1
        print(f"Parsing completed with {error_count} errors")


if __name__ == '__main__':
    parse_piqa = ParsePiqa()
    parse_piqa.setup()

