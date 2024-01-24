import argparse

import numpy as np

from tools.utils import *
from data.high_gamma_process import load_highgamma_data_single_subject
from data.p300_process import load_p300_data_single_subject
from data.physionet_process import load_physionet_data_single_subject
from data.eegdataset import EEGDataset
import torch as th
from torch.utils.data import DataLoader
from models.EEGNet import EEGNet
import time
import datetime


def main(args):
    # ----------------------------------------------environment setting-----------------------------------------------
    set_seed(args.seed)
    args = set_save_path(args.father_path, args, time_frame=False)
    sys.stdout = Logger(os.path.join(args.log_path, 'information.txt'))
    # ------------------------------------------------device setting--------------------------------------------------
    device = 'cuda:' + str(args.cuda) if th.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # ------------------------------------------------data setting----------------------------------------------------

    if 'high_gamma' in args.data_path:
        load_data = load_highgamma_data_single_subject
        args.sub_num = 14
        args.class_num = 4
    elif 'physionet' in args.data_path:
        load_data = load_physionet_data_single_subject
        args.sub_num = 10
        args.class_num = 4
    elif 'P300' in args.data_path:
        args.sub_num = 8
        args.class_num = 2
        load_data = load_p300_data_single_subject
    else:
        raise ValueError('Wrong data path!')

    X_1_list, y_1_list = [], []
    v_X_1_list, v_y_1_list = [], []

    for i in range(args.sub_num):
        if i + 1 != args.tid:
            X_1, y_1, v_X_1, v_y_1 = load_data(args.data_path, subject_id=i + 1, to_tensor=False)
            X_1_list.append(X_1)
            y_1_list.append(y_1)
            v_X_1_list.append(v_X_1)
            v_y_1_list.append(v_y_1)

    X_1 = np.concatenate(X_1_list, axis=0)
    y_1 = np.concatenate(y_1_list, axis=0)
    v_X = np.concatenate(v_X_1_list, axis=0)
    v_y = np.concatenate(v_y_1_list, axis=0)
    train_data = EEGDataset(None, X_1, None, y_1)

    trainLoader = DataLoader(train_data,
                             batch_size=args.batch_size,
                             num_workers=4,
                             shuffle=True,
                             drop_last=False)

    test_data = EEGDataset(None, v_X, None, v_y)

    # '''
    # ------------------------------------------------model setting---------------------------------------------------
    model = EEGNet(X_1.shape[1], X_1.shape[2], args.class_num, drop_prob=args.dropout,
                   f1=args.f1, d=2, f2=args.f1 * 2, kernel_length=64, classifier=True).to(device)
    opt = th.optim.Adam(filter(lambda x: x.requires_grad is not False, model.parameters()), lr=args.lr,
                        weight_decay=args.w_decay)
    cls_criterion = th.nn.CrossEntropyLoss().to(device)
    stop_train = EarlyStopping(patience=30, max_epochs=args.epochs)
    start_time = time.time()
    best_acc = 0.0

    # ------------------------------------------------training---------------------------------------------------
    model.to(device)
    for epoch in range(args.epochs):
        if stop_train.early_stop:
            print("early stop in {}!".format(epoch))
            break
        print('--------------------------Start training At Epoch:{}--------------------------'.format(epoch + 1))
        model.train()
        dict_log = {'loss': AverageMeter(), 'cls_acc': AverageMeter()}
        for step, (X, labels) in enumerate(trainLoader):
            X = X.to(device)
            labels = labels[:, 0].to(device)
            f, logits = model(X)
            loss = cls_criterion(logits, labels)
            dict_log['loss'].update(loss.item(), len(X))
            cls_acc = accuracy(logits.detach(), labels.detach())[0]
            dict_log['cls_acc'].update(cls_acc.item(), len(X))
            if not math.isfinite(loss.item()):
                print("Loss is {} at step{}/{}, stopping training.".format(loss.item(), step, epoch))
                print(loss.item())
                sys.exit(1)
            opt.zero_grad()
            loss.backward()
            opt.step()

        lr = list(opt.param_groups)[0]['lr']
        now_time = time.time() - start_time
        et = str(datetime.timedelta(seconds=now_time))[:-7]

        print_information = 'epoch:{}/{}\ttime consumption:{}\tlr:{}\n'.format(
            epoch + 1, args.epochs, et, lr)

        cls_loss_info = "loss_ent: (val/avg):{:.3f}/{:.3f}\t".format(dict_log['loss'].val, dict_log['loss'].avg)
        print_information += cls_loss_info
        cls_acc_info = "cls_acc: (val/avg):{:.3f}/{:.3f}\n".format(dict_log['cls_acc'].val, dict_log['cls_acc'].avg)
        print_information += cls_acc_info
        print(print_information)
        print('--------------------------End training At Epoch:{}--------------------------'.format(epoch + 1))
        print('--------------------------Start Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
        dict_log = {'loss': AverageMeter(), 'acc': AverageMeter()}

        model.eval()
        testLoader = DataLoader(test_data,
                                batch_size=args.batch_size,
                                num_workers=4,
                                drop_last=False)
        for step, (X1, labels) in enumerate(testLoader):
            X1 = X1.to(device)
            labels = labels[:, 0].to(device)
            _, logits = model(X1)
            loss = cls_criterion(logits, labels)
            dict_log['loss'].update(loss.item(), len(X1))
            acc = accuracy(logits.detach(), labels.detach())[0]
            dict_log['acc'].update(acc.item(), len(X1))

        now_time = time.time() - start_time
        et = str(datetime.timedelta(seconds=now_time))[:-7]

        print_information = 'epoch:{}/{}\ttime consumption:{}\n'.format(epoch + 1, args.epochs, et)

        loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('all_loss',
                                                         dict_log['loss'].val, dict_log['loss'].avg)
        print_information += loss_info
        acc_info = "acc:(val/avg):{:.3f}/{:.3f}\t".format(dict_log['acc'].val, dict_log['acc'].avg)
        print_information += (acc_info + '\n')
        print(print_information)
        print('--------------------------End Evaluate At Epoch:{}--------------------------'.format(epoch + 1))
        current_acc = dict_log['acc'].avg

        if current_acc > best_acc:
            best_acc = current_acc
            save({
                'epoch': epoch + 1,
                'model_classifier': model.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(args.log_path, 'model_best.pth.tar'))
        print('best_acc:{} at epoch{}'.format(best_acc, epoch + 1))
        save({
            'epoch': epoch + 1,
            'model_classifier': model.state_dict(),
            'best_acc': best_acc,
        }, os.path.join(args.log_path, 'model_newest.pth.tar'))
        stop_train(current_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str,
                        default='data/preprocess/physionet',
                        help='Data path.')
    parser.add_argument('-f1', type=int, default=32,
                        help='the number of filters in EEGNet.')
    parser.add_argument('-tid', type=int, default=11, help='Target id. If data is physionet, pls set tid >= 11')
    parser.add_argument('-dropout', type=float, default=0.25, help='Dropout rate for downstream model.')
    parser.add_argument('-epochs', type=int, default=250, help='Number of epochs to train.')
    parser.add_argument('-cuda', type=int, default=0, help='cuda')
    parser.add_argument('-lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('-w_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-batch_size', default=128, type=int,
                        help='Batch size.')
    parser.add_argument('-print_freq', type=int, default=5, help='The frequency to show training information.')
    parser.add_argument('-father_path', type=str, default='save/physionet',
                        help='The father path of models parameters, log files.')
    parser.add_argument('-seed', type=int, default='111', help='Random seed.')
    args_ = parser.parse_args()
    main(args_)
    # visualization(args_)
