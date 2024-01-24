import os
import torch
import pickle


def load_p300_data_single_subject(filename, subject_id, to_tensor=True):
    subject_id = str(subject_id)
    train_path = os.path.join(filename, 'P300_train.pkl')
    test_path = os.path.join(filename, 'P300_test.pkl')
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    try:
        subject_id = '0' + subject_id if int(subject_id) < 10 else subject_id
        train_X, train_y = train_data[subject_id]['X'], train_data[subject_id]['y']
        test_X, test_y = test_data[subject_id]['X'], test_data[subject_id]['y']
    except KeyError:
        subject_id = int(subject_id)
        train_X, train_y = train_data[subject_id]['X'], train_data[subject_id]['y']
        test_X, test_y = test_data[subject_id]['X'], test_data[subject_id]['y']
    if to_tensor:
        train_X = torch.tensor(train_X, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.int64)
        test_y = torch.tensor(test_y, dtype=torch.int64)
    return train_X, train_y, test_X, test_y
