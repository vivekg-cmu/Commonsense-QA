import sys

sys.path.append(".")

from torch.utils import data
import numpy as np
zzz
class PretrainingDataset(data.Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __len__(self):
        return len(self.input_dict["input_ids"])

    def __getitem__(self, index):
        input_ids = np.array(self.input_dict["input_ids"][index])
        input_mask = np.array(self.input_dict["input_mask"][index])
        segment_ids = np.array(self.input_dict["segment_ids"][index])
        masked_lm_positions = np.array(self.input_dict["masked_lm_positions"][index])
        masked_lm_ids = np.array(self.input_dict["masked_lm_ids"][index])
        masked_lm_weights = np.array(self.input_dict["masked_lm_weights"][index])
        next_sentence_labels = np.array(self.input_dict["next_sentence_labels"][index])

        return input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids,\
               masked_lm_weights, next_sentence_labels