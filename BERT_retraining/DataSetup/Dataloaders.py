from torch.utils import data


class PretrainingDataset(data.Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __len__(self):
        return len(self.input_dict[self.input_dict[0]])

    def __getitem__(self, index):
        input_ids = self.input_dict["input_ids"][index]
        input_mask = self.input_dict["input_mask"][index]
        segment_ids = self.input_dict["segment_ids"][index]
        masked_lm_positions = self.input_dict["masked_lm_positions"][index]
        masked_lm_ids = self.input_dict["masked_lm_ids"][index]
        masked_lm_weights = self.input_dict["masked_lm_weights"][index]
        next_sentence_labels = self.input_dict["next_sentence_labels"][index]

        return input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids,\
               masked_lm_weights, next_sentence_labels


if __name__ == '__main__':
    from BERT_retraining import utils

    loaded_features = utils.load_dictionary(
        "/home/pratik/Desktop/new_github/Commonsense-QA/BERT_retraining/Data/features.pkl")
    pretrain_loader = PretrainingDataset(input_dict=loaded_features)