import sys

sys.path.append(".")

from torch.utils import data
import numpy as np


class DownstreamDataset(data.Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __len__(self):
        return len(self.input_dict["label"])

    def __getitem__(self, index):
        ans_a = np.array(self.input_dict["ans_a"][index])
        ans_b = np.array(self.input_dict["ans_b"][index])
        ans_c = np.array(self.input_dict["ans_c"][index])
        ans_a_att = np.array(self.input_dict["ans_a_att"][index])
        ans_b_att = np.array(self.input_dict["ans_b_att"][index])
        ans_c_att = np.array(self.input_dict["ans_c_att"][index])
        ans_a_token = np.array(self.input_dict["ans_a_token"][index])
        ans_b_token = np.array(self.input_dict["ans_b_token"][index])
        ans_c_token = np.array(self.input_dict["ans_c_token"][index])
        label = np.array(self.input_dict['label'][index])
        return ans_a, ans_b, ans_c, ans_a_att, ans_b_att, ans_c_att, \
               ans_a_token, ans_b_token, ans_c_token, label
