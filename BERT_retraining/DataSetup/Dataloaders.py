import sys

sys.path.append(".")

from torch.utils import data
import numpy as np


class DownstreamDataset(data.Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __len__(self):
        return len(self.input_dict["labels"])

    def __getitem__(self, index):
        ans_a = np.array(self.input_dict["ans_a"][index])
        ans_b = np.array(self.input_dict["ans_a"][index])
        ans_c = np.array(self.input_dict["ans_a"][index])
        label = np.array(self.input_dict['label'][index])
        return ans_a, ans_b, ans_c, label
