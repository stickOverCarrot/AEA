import os
import torch
import pickle
import scipy.io as sio
import numpy as np


def load_physionet_data_single_subject(filename, subject_id, to_tensor=True):
    # subject_id = str(subject_id)
    train_path = os.path.join(filename, 'physionet_train.pkl')
    test_path = os.path.join(filename, 'physionet_test.pkl')
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    try:
        train_X, train_y = train_data[subject_id]['X'], train_data[subject_id]['y']
        test_X, test_y = test_data[subject_id]['X'], test_data[subject_id]['y']
    except KeyError:
        subject_id = str(subject_id)
        train_X, train_y = train_data[subject_id]['X'], train_data[subject_id]['y']
        test_X, test_y = test_data[subject_id]['X'], test_data[subject_id]['y']
    if to_tensor:
        train_X = torch.tensor(train_X, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.int64)
        test_y = torch.tensor(test_y, dtype=torch.int64)
    return train_X, train_y, test_X, test_y


def process_physionet(mat_path, pkl_path):
    labels = sio.loadmat(os.path.join(mat_path, 'Labels_1.mat'))['Labels']
    labels = np.argmax(labels, axis=-1)
    dd = np.zeros([20, 84, 640, 64])
    for i in range(1, 65):
        train_data = sio.loadmat(os.path.join(mat_path, 'Dataset_{}.mat'.format(i)))
        dd[:, :, :, i - 1] = train_data['Dataset']
        # dd = dd.transpose([0, 1, 3, 2])

    for i in range(20):
        for j in range(64):
            dd[i, :, :, j] = dd[i, :, :, j] - np.mean(dd[i, :, :, j])

    tt = np.zeros([20, 84, 640, 64])
    for i in range(20):
        for j in range(64):
            for k in range(640):
                mean_x = dd[i, :, k, j].mean()
                std_x = dd[i, :, k, j].std()
                tt[i, :, k, j] = (dd[i, :, k, j] - mean_x) / std_x

    tt = tt.transpose([0, 1, 3, 2])
    length = 14
    train_data = {}
    test_data = {}
    for i in range(20):
        y = labels[i]
        x = tt[i]
        # shuffle
        index = np.arange(len(y))
        np.random.shuffle(index)
        x = x[index]
        y = y[index]
        train_data[str(i + 1)] = {'X': [], 'y': []}
        test_data[str(i + 1)] = {'X': [], 'y': []}
        for j in range(4):
            x_ = x[y == j]
            y_ = y[y == j]
            train_x = x_[:length]
            train_y = y_[:length]
            test_x = x_[length:]
            test_y = y_[length:]
            train_data[str(i + 1)]['X'].append(train_x)
            train_data[str(i + 1)]['y'].append(train_y)
            test_data[str(i + 1)]['X'].append(test_x)
            test_data[str(i + 1)]['y'].append(test_y)
        train_data[str(i + 1)]['X'] = np.concatenate(train_data[str(i + 1)]['X'], axis=0)
        train_data[str(i + 1)]['y'] = np.concatenate(train_data[str(i + 1)]['y'], axis=0)
        test_data[str(i + 1)]['X'] = np.concatenate(test_data[str(i + 1)]['X'], axis=0)
        test_data[str(i + 1)]['y'] = np.concatenate(test_data[str(i + 1)]['y'], axis=0)

    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    with open(os.path.join(pkl_path, 'physionet_train.pkl'), 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(pkl_path, 'physionet_test.pkl'), 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    process_physionet('20-Subjects', 'preprocess/physionet')
