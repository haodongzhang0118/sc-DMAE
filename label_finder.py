import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    data_mat = h5py.File(
        f"{path}.h5", "r")
    X = np.array(data_mat['X'])
    Y = np.array(data_mat['Y'])
    encoder_x = LabelEncoder()
    Y = encoder_x.fit_transform(Y)
    int_to_label = {idx: label for idx, label in enumerate(encoder_x.classes_)}

    return int_to_label


data_path = '/Users/anthony/Downloads/Pollen'
int_to_label = load_data(data_path)

print(int_to_label)