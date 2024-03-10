import pickle


def save_pickle(object, path):
    with open(path, "wb") as f:
        pickle.dump(object, f)

    return


def load_pickle(path):
    with open(path, "rb") as f:
        object = pickle.load(f)

    return object
