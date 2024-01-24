import torch as th
from torch.utils.data import Dataset
from sklearn.utils import check_random_state
import numpy as np
import random


class EEGDataset(Dataset):
    def __init__(self, source_X_list, target_X, source_y_list, target_y, data_num=None, pds=None):
        super(EEGDataset, self).__init__()
        self.data_num = data_num
        self.t1 = t1
        self.t2 = t2
        if source_X_list is not None:
            self.num_cls = len(set(source_y_list[0]))
            self.length = sum([len(source_X_list[i]) for i in range(len(source_X_list))])
            self.num_source = len(source_X_list)
            if target_X is not None:
                self.X = np.concatenate([np.concatenate(source_X_list), target_X])
                self.y = np.concatenate([np.concatenate(source_y_list), target_y])
                self.sd = self.set_sd(source_X_list + [target_X])
                self.length = len(self.X)
            else:
                self.X = np.concatenate(source_X_list)
                self.y = np.concatenate(source_y_list)
                self.sd = self.set_sd(source_X_list)
        elif target_X is not None:
            self.num_cls = len(set(target_y))
            self.length = len(target_X)
            self.X = target_X
            self.y = target_y
            self.sd = self.set_sd([target_X])
        else:
            raise ValueError("It is at least one of source_X_list and target_X is not None")
        self.sd = np.asarray(self.sd, dtype=np.int64)
        self.index = np.arange(len(self.X)).astype(np.int64)
        if pds is not None:
            self.pds = pds
            self.info = np.vstack([self.y, self.sd, self.pds, self.index]).T
        else:
            self.info = np.vstack([self.y, self.sd, self.index]).T

    def __len__(self):
        if self.data_num is not None:
            return self.data_num
        return self.length

    def __getitem__(self, item):
        return th.FloatTensor(self.X[item]), th.LongTensor(self.info[item])

    def set_sd(self, X):
        sd = []
        for i in range(len(X)):
            sd.extend([i+1] * len(X[i]))
        return sd


