import argparse

from tools.utils import *
from data.high_gamma_process import load_highgamma_data_single_subject
from data.p300_process import load_p300_data_single_subject
from data.physionet_process import load_physionet_data_single_subject
from data.eegdataset import EEGDataset
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.EEGNet import EEGNetP
import time
import datetime
from sklearn import metrics


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

    X_1, y_1, v_X_1, v_y_1 = load_data(args.data_path, subject_id=args.tid, to_tensor=False)
    if 'P300' in args.data_path:
        X_1, y_1 = sample_data(X_1, y_1, 500)

    train_data = EEGDataset(None, X_1, None, y_1)
    trainLoader = DataLoader(train_data,
                             batch_size=args.batch_size,
                             num_workers=4,
                             shuffle=True,
                             drop_last=False)

    test_data = EEGDataset(None, v_X_1, None, v_y_1)
    # ------------------------------------------------model setting---------------------------------------------------

    model = EEGNetP(X_1.shape[1], X_1.shape[2], args.class_num, drop_prob=args.dropout,
                    f1=args.f1, d=2, f2=args.f1 * 2, kernel_length=64, classifier=True).to(device)

    print(args)

    load_checkpoints = th.load(os.path.join(args.resume_path, 'model_best.pth.tar'))
    model_dict = model.state_dict()
    T1 = {}
    for key, values in load_checkpoints['model_classifier'].items():
        if key in model_dict.keys():
            if key.startswith('weight') or key.startswith('b') or key.startswith('A'):
                continue
            T1[key] = values
    model_dict.update(T1)
    model.load_state_dict(model_dict)

    for name, p in model.named_parameters():
        if name.startswith('weight') or name.startswith('b') or name.startswith('A'):
            continue
        p.requires_grad = False

    for name, p in model.named_parameters():
        if name.startswith('cls'):
            p.requires_grad = False

    opt = th.optim.Adam(filter(lambda x: x.requires_grad is not False, model.parameters()), lr=args.lr,
                        weight_decay=0)
    cls_criterion = th.nn.CrossEntropyLoss().to(device)
    stop_train = EarlyStopping(patience=80, max_epochs=args.epochs)
    start_time = time.time()
    best_acc = 0.0

    # ------------------------------------------------training---------------------------------------------------
    model.to(device)
    num_sample = len(trainLoader.dataset)
    fea_bank = th.randn(num_sample, model.get_feature_dim())
    score_bank = th.randn(num_sample, args.class_num).to(device)
    model.eval()
    with th.no_grad():
        iter_test = iter(trainLoader)
        for i in range(len(trainLoader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[1][:, -1]
            # labels = data[1][:, 0]
            inputs = inputs.to(device)
            output, outputs = model(inputs)
            output_norm = F.normalize(output.reshape(output.shape[0], -1), p=2, dim=1)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()

    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter()}
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

    print_information = 'epoch:{}/{}\ttime consumption:{}\n'.format(0, args.epochs, et)

    loss_info = "{}(val/avg):{:.3f}/{:.3f}\n".format('all_loss',
                                                     dict_log['loss'].val, dict_log['loss'].avg)
    print_information += loss_info
    acc_info = "acc:(val/avg):{:.3f}/{:.3f}\t".format(dict_log['acc'].val, dict_log['acc'].avg)
    print_information += (acc_info + '\n')
    print(print_information)

    for epoch in range(args.epochs):
        if stop_train.early_stop:
            print("early stop in {}!".format(epoch))
            break
        print('--------------------------Start training At Epoch:{}--------------------------'.format(epoch + 1))
        model.train()
        # model.apply(set_bn_eval) # freeze BN will cause the performance drop slightly for hgd and physionet, but work better for P300
        dict_log = {'im_div': AverageMeter(), 'loss_const': AverageMeter(), 'cls_acc': AverageMeter()}

        labels_list = []
        tar_idx = []
        features_test = []
        logits_list = []
        # calculate the gradient on the whole training set to get the better performance
        for step, (X, infos) in enumerate(trainLoader):
            X = X.to(device)
            labels = infos[:, 0].to(device)
            indx = infos[:, -1]
            output_f, logits = model(X)
            output_f = output_f.reshape(output_f.shape[0], -1)
            labels_list.append(labels)
            tar_idx.append(indx)
            features_test.append(output_f)
            logits_list.append(logits)

        labels = th.cat(labels_list, dim=0)
        tar_idx = th.cat(tar_idx, dim=0)
        features_test = th.cat(features_test, dim=0)
        logits = th.cat(logits_list, dim=0)
        softmax_out = nn.Softmax(dim=1)(logits)

        with th.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = th.topk(distance,
                                  dim=-1,
                                  largest=True,
                                  k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C

            fea_near = fea_bank[idx_near]  # batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
            distance_ = th.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
            _, idx_near_near = th.topk(distance_, dim=-1, largest=True,
                                       k=args.KK + 1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (
                    idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = th.where(
                match > 0., match,
                th.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    args.KK)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            # weight_kk[idx_near_near == tar_idx_]=0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            # print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                            args.class_num)  # batch x KM x C

            score_self = score_bank[tar_idx]

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK,
                                                    -1)  # batch x C x 1
        const = th.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.to(device)).sum(
                1))  # kl_div here equals to dot product since we do not use log for score_near_kk
        loss = th.mean(const)

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K,
                                                         -1)  # batch x K x C

        loss += th.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.to(device)).sum(1))
        dict_log['loss_const'].update(loss.item(), len(X))
        # self, if not explicitly removing the self feature in expanded neighbor then no need for this
        # loss += -torch.mean((softmax_out * score_self).sum(-1))

        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = th.sum(msoftmax *
                               th.log(msoftmax + 1e-5))
        loss += gentropy_loss

        if not math.isfinite(loss.item()):
            print("Loss is {} at epoch{}, stopping training.".format(loss.item(), epoch))
            print(loss.item())
            sys.exit(1)
        opt.zero_grad()
        loss.backward()
        # th.nn.utils.clip_grad_norm(model.parameters(), 10000)
        opt.step()

        dict_log['im_div'].update(gentropy_loss.item(), len(X))
        cls_acc = accuracy(logits.detach(), labels.detach())[0]
        dict_log['cls_acc'].update(cls_acc.item(), len(X))
        lr = list(opt.param_groups)[0]['lr']
        now_time = time.time() - start_time
        et = str(datetime.timedelta(seconds=now_time))[:-7]

        print_information = 'epoch:{}/{}\ttime consumption:{}\tlr:{}\n'.format(
            epoch + 1, args.epochs, et, lr)

        cls_loss_info = "im_div: (val/avg):{:.3f}/{:.3f}\t".format(dict_log['im_div'].val, dict_log['im_div'].avg)
        print_information += cls_loss_info
        cls_loss_info = "loss_const: (val/avg):{:.3f}/{:.3f}\t".format(dict_log['loss_const'].val,
                                                                       dict_log['loss_const'].avg)
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
        pred_list = []
        labels_list = []
        for step, (X1, labels) in enumerate(testLoader):
            X1 = X1.to(device)
            labels = labels[:, 0].to(device)
            weight, logits = model(X1)
            loss = cls_criterion(logits, labels)
            dict_log['loss'].update(loss.item(), len(X1))
            acc = accuracy(logits.detach(), labels.detach())[0]
            dict_log['acc'].update(acc.item(), len(X1))
            pred = th.argmax(logits, dim=1)
            pred_list.append(pred.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
        pred_list = np.concatenate(pred_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)
        kappa_ = metrics.cohen_kappa_score(labels_list, pred_list)

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
                'best_kappa:': kappa_,
            }, os.path.join(args.log_path, 'model_best.pth.tar'))
        print('best_acc:{} at epoch{}'.format(best_acc, epoch + 1))
        save({
            'epoch': epoch + 1,
            'model_classifier': model.state_dict(),
            'best_acc': best_acc,
            'best_kappa:': kappa_,
        }, os.path.join(args.log_path, 'model_newest.pth.tar'))
        stop_train(current_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str,
                        default='data/preprocess/physionet',
                        help='Data path.')
    parser.add_argument('-f1', type=int, default=32,
                        help='the number of filters in EEGNet.')
    parser.add_argument('-tid', type=int, default=1, help='Target id. If data is physionet, pls set tid >= 11')
    parser.add_argument('-dropout', type=float, default=0., help='Dropout rate for downstream model.')
    parser.add_argument('-epochs', type=int, default=250, help='Number of epochs to train.')
    parser.add_argument('-cuda', type=int, default=0, help='cuda')
    parser.add_argument('-lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('-adjust_lr', type=int, default=0, choices=[0, 1, 2], help='Learning rate changes over epoch.')
    parser.add_argument('-w_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-batch_size', default=56, type=int,
                        help='Batch size.')
    parser.add_argument('-print_freq', type=int, default=5, help='The frequency to show training information.')
    parser.add_argument('-father_path', type=str, default='save_AEA/physionet',
                        help='The father path of models parameters, log files.')
    parser.add_argument('-seed', type=int, default='111', help='Random seed.')
    parser.add_argument('-resume_path', type=str,
                        default='save/physionet/11', help='Path of saved model.')
    args_ = parser.parse_args()
    main(args_)
