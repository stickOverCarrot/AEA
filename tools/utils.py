import numpy as np
import torch as th
import random
import errno
import os
import sys
import time
import math
import pickle
from sklearn.utils import check_random_state


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_save_path(father_path, args, time_frame=True):
    if time_frame:
        father_path = os.path.join(father_path, '{}'.format(time.strftime("%m_%d_%H_%M")))
    else:
        father_path = os.path.join(father_path, "{}".format(args.tid))
    # try:
    #     father_path = os.path.join(father_path, args.model + '_{}'.format(args.f1) + '_{}'.format(args.dg),
    #                                str(args.target_id))
    # except:
    #     father_path = os.path.join(father_path, args.model + '_{}'.format(args.f1), str(args.target_id))
    mkdir(father_path)
    args.log_path = father_path
    args.model_classifier_path = father_path
    return args


def save(checkpoints, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    th.save(checkpoints, save_path)


def accuracy(output, target, topk=(1,)):
    shape = None
    if 2 == len(target.size()):
        shape = target.size()
        target = target.view(target.size(0))
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    if shape:
        target = target.view(shape)
    return ret


class Logger(object):
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class Logger_t(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_cls_entropy(probability):
    preds = th.stack(probability).permute(1, 0, 2)
    entropy = -preds * th.log(preds)
    entropy = entropy.sum(dim=-1)
    min_entropy_loc = th.argmin(entropy, dim=-1)
    idx = th.LongTensor([i for i in range(len(preds))])
    return preds[idx, min_entropy_loc]


def lam_change_over_epoch(epoch, all_epoch, min_value=None, max_value=None):
    if min_value is not None and max_value is None:
        assert min_value <= max_value
    p = epoch / all_epoch
    value = 2 / (1 + math.exp(-10 * p)) - 1
    # value = value/2
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(min_value, value)
    return value


def lr_change_over_epoch1(opt, init_value, epoch, all_epoch, min_value=None, max_value=None):
    if min_value is not None and max_value is None:
        assert min_value <= max_value
    p = epoch / all_epoch
    value = init_value / math.pow(1 + 10 * p, 0.75)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(min_value, value)
    for param in opt.param_groups:
        param['lr'] = value


def lr_change_over_epoch2(opt, init_value, epoch, epochs):
    # epoch = epoch + 1
    #     # if epoch <= 5:
    #     #     lr = init_value * epoch / 5
    #     # elif epoch > 100:
    #     #     lr = init_value * 0.1
    #     # elif epoch > 200:
    #     #     lr = init_value * 0.1
    #     # else:
    #     #     lr = init_value
    if epoch < epochs / 5:
        lr = 0.001 * 5 * epoch / epochs
    else:
        lr = 0.001 * 0.5 * (1 + math.cos(180 * (epoch - epochs / 5) / (epochs / 4)))
    for param_group in opt.param_groups:
        param_group['lr'] = lr


class EarlyStopping(object):
    """
    Early stops the training if validation loss
    doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, max_epochs=80):
        """
        patience (int): How long to wait after last time validation
        loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation
        loss improvement.
                        Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        # my addition:
        self.max_epochs = max_epochs
        self.max_epoch_stop = False
        self.epoch_counter = 0
        self.should_stop = False
        self.checkpoint = None

    def __call__(self, val_loss):
        # my addition:
        self.epoch_counter += 1
        if self.epoch_counter >= self.max_epochs:
            self.max_epoch_stop = True

        score = val_loss

        if self.best_score is None:
            print('')
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} '
                  f'out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        # my addition:
        if any([self.max_epoch_stop, self.early_stop]):
            self.should_stop = True


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def add_noise_with_SNR(signals, noise_amount=20):
    """
    adding noise
    created using: https://stackoverflow.com/a/53688043/10700812
    """
    new_signals = []
    for signal in signals:
        target_snr_db = noise_amount  # 20
        x_watts = signal ** 2  # Calculate signal power and convert to dB
        sig_avg_watts = np.mean(x_watts, axis=1)
        sig_avg_db = 10 * np.log10(sig_avg_watts)  # Calculate noise then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts.mean()),
                                       x_watts.shape)  # Generate an sample of white noise
        noised_signal = signal + noise_volts  # noise added signal
        new_signals.append(noised_signal)
    new_signals = np.asarray(new_signals)
    return new_signals


def sample_data(X, y, num_per_class=42):
    new_X = []
    new_y = []
    for i in range(len(set(y))):
        x_ = X[i == y][:num_per_class]
        y_ = y[i == y][:num_per_class]
        new_X.append(x_)
        new_y.append(y_)
    new_X = np.concatenate(new_X, axis=0)
    new_y = np.concatenate(new_y, axis=0)
    return new_X, new_y


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * th.log(input_ + epsilon)
    entropy = th.sum(entropy, dim=1)
    return entropy