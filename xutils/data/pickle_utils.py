import pickle


def write(file_name, data, protocol=pickle.HIGHEST_PROTOCOL):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=protocol)


def read(file_name):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)
