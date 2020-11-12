import sys

sys.path.append(".")

from torch.utils import data
import numpy as np


class DownstreamDataset(data.Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __len__(self):
        return len(self.input_dict["A"])

    def __getitem__(self, index):
        ans_a = np.array(self.input_dict["A"][index])
        ans_b = np.array(self.input_dict["B"][index])
        ans_c = np.array(self.input_dict["C"][index])
        ans_d = np.array(self.input_dict["D"][index])
        ans_e = np.array(self.input_dict["E"][index])

        ans_a_att = np.array(self.input_dict["A_att"][index])
        ans_b_att = np.array(self.input_dict["B_att"][index])
        ans_c_att = np.array(self.input_dict["C_att"][index])
        ans_d_att = np.array(self.input_dict["D_att"][index])
        ans_e_att = np.array(self.input_dict["E_att"][index])

        label = np.array(self.input_dict['label'][index])
        return ans_a, ans_b, ans_c, ans_d, ans_e, ans_a_att, \
               ans_b_att, ans_c_att, ans_d_att, ans_e_att, label
