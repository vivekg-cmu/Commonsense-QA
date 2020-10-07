import pickle


def save_dictionary(dictionary, save_path):
    with open(save_path, 'wb') as handle:
        print("saving model to:", save_path)
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("dictionary saved to:", save_path)


def load_dictionary(load_path):
    with open(load_path, 'rb') as handle:
        print("loading data from:", load_path)
        dictionary = pickle.load(handle)
        print("loading completed")
        return dictionary
