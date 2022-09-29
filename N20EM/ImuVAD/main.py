import os, random, time, sys
import numpy as np
import pandas as pd
import time

from tqdm import tqdm
from models import *

import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from PIL import Image as Img
from utils import *

from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

random.seed(10)
np.random.seed(0)
torch.manual_seed(10)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

hparam = {
    'check_only': False,

    'dataset_root': '/data1/anonymous/n20em/dataset_v1.0/IMU_VAD_data',
    'log_dir': 'train',
    'use_tensorboard': True,

    'MODEL_NAME': 'IMU_CRNN_Ott_GRU_3',
    'DROPOUT_CNN': 0.5,
    'DROPOUT_RNN': 0.2,
    'RNN_WIDTH': 60,

    'EPOCH': 50,
    'BATCH_SIZE': 128,
    'LR': 1e-3,
    'WEIGHT_DECAY': 1e-3,
    'REDUCE_EPOCH': 2,
    'REDUCE_FACTOR': 0.9,

    'EARLY_STOP': True,
    'PATIENCE': 1e4,

    'device': "cuda",
}


def main():
    train(hparam['MODEL_NAME'])


def get_model(model_name):
    if model_name in ['IMU_CRNN_Ott', 'IMU_CRNN_Ott_1', 'IMU_CRNN_Ott_2', 'IMU_CRNN_Ott_4', 'IMU_CRNN_Ott_GRU',
                      'IMU_CRNN_Ott_GRU_2',
                      'IMU_CRNN_Ott_GRU_3', 'IMU_CRNN_Ott_GRU_4', 'IMU_CRNN_Ott_GRU_5']:
        M = eval(model_name)
        model = M(
            dropout_cnn=hparam['DROPOUT_CNN'],
            dropout_rnn=hparam['DROPOUT_RNN'],
            rnn_width=hparam['RNN_WIDTH']
        )
    else:
        raise NotImplementedError
    return model


def train(model_name, ctn=False):
    log_dir = jpath('log', model_name, hparam['log_dir'])
    if not os.path.exists(jpath('log', model_name)):
        os.mkdir(jpath('log', model_name))
    if hparam['use_tensorboard']:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        else:
            raise Exception('Log already exists!')
        save_json(hparam, jpath(log_dir, 'hparam.json'))
        writer = SummaryWriter(jpath(log_dir, 'tensorboard'))

    if not os.path.exists('./ckpt/' + model_name):
        os.mkdir('./ckpt/' + model_name, )

    dataset_train = MyDataset(hparam['dataset_root'], split='train')
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=hparam['BATCH_SIZE'],
        shuffle=True,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )

    dataset_valid = MyDataset(hparam['dataset_root'], split='valid')
    valid_loader = DataLoader(
        dataset=dataset_valid,
        batch_size=hparam['BATCH_SIZE'],
        shuffle=False,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )

    dataset_test = MyDataset(hparam['dataset_root'], split='test')
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=hparam['BATCH_SIZE'],
        shuffle=False,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Model, optimizer, loss function, earlystopping
    net = get_model(model_name).to(hparam['device'])
    # if ctn == True:
    #     net.load_state_dict(torch.load('models/' + model_name + '.pth'))
    print(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=hparam['LR'], weight_decay=hparam['WEIGHT_DECAY'])
    loss_func = nn.BCELoss()
    early_stopping = EarlyStopping(hparam['PATIENCE'], verbose=False, savefolder=log_dir)

    param_num = check_model(net)
    if hparam['check_only'] == True:
        return

    with open(jpath(log_dir, 'net.txt'), 'w') as f:
        f.write('{}'.format(net))
        f.write('Num of param: {}'.format(param_num))

    # Training and validating
    time_begin = time.time()
    for epoch in range(hparam['EPOCH']):


        net.train()
        running_loss = 0.0
        step_cnt = 0
        corr_cnt = 0
        running_macro_f1 = 0.0
        running_f1 = 0.0
        pbar = tqdm(train_loader)
        for step, (b_x, b_y) in enumerate(pbar):
            b_x = b_x.to(hparam['device'])
            b_y = b_y.to(hparam['device'])
            # print(step)

            # print(b_x.shape)
            out = net(b_x)
            # print(out.shape)
            # print(out.shape, b_y.shape)
            loss = loss_func(out, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # step_cnt += 1

            # training acc
            pred_y = compute_pred(out).flatten()
            target_y = b_y.data.cpu().numpy().flatten()
            accuracy = sum(pred_y == target_y) / pred_y.shape[0]
            corr_cnt += accuracy

            macro_f1 = compute_macro_f1(target_y, pred_y)
            running_macro_f1 += macro_f1

            f1 = compute_f1(target_y, pred_y)
            running_f1 += f1

            # print('Epoch: {} | Step: {} / {} | Accuracy: {:.4f} | Loss: {:.4f}'.format(
            #     epoch+1, step+1, len(train_loader), accuracy, loss.item()
            # ))
            pbar.set_description('Epoch: {} | Step {} / {}'.format(epoch + 1, step + 1, len(train_loader)))

        if epoch % hparam['REDUCE_EPOCH'] == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= hparam['REDUCE_FACTOR']

        avg_train_loss = running_loss / len(train_loader)
        train_acc = corr_cnt / len(train_loader)
        avg_train_macro_f1 = running_macro_f1 / len(train_loader)
        train_f1 = running_f1 / len(train_loader)
        print('Epoch: {} | Training Loss: {:.4f} | Acc: {:.4f} | F1: {:.4f}, {:.4f}'.format(epoch + 1, avg_train_loss,
                                                                                            train_acc, train_f1,
                                                                                            avg_train_macro_f1),
              end='')

        # Validating
        running_loss = 0.0
        # step_cnt = 0
        corr_cnt = 0
        running_f1 = 0.0
        running_macro_f1 = 0.0
        net.eval()
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(valid_loader):
                out = net(batch_x.to(hparam['device']))
                loss = loss_func(out, batch_y.to(hparam['device']))
                running_loss += loss.item()
                step_cnt += 1

                pred = compute_pred(out).flatten()
                batch_y = batch_y.data.cpu().numpy().flatten()
                accuracy = sum(pred == batch_y) / pred.shape[0]
                corr_cnt += accuracy

                macro_f1 = compute_macro_f1(batch_y, pred)
                running_macro_f1 += macro_f1

                f1 = compute_f1(batch_y, pred)
                running_f1 += f1

        valid_acc = corr_cnt / len(valid_loader)
        avg_valid_loss = running_loss / len(valid_loader)
        avg_valid_macro_f1 = running_macro_f1 / len(valid_loader)
        valid_f1 = running_f1 / len(valid_loader)
        print(
            ' | Validation Loss: {:.4f} | Acc: {:.4f} | F1: {:.4f}, {:.4f}'.format(avg_valid_loss, valid_acc, valid_f1,
                                                                                   avg_valid_macro_f1), end='')

        # Validating by test data
        running_loss = 0.0
        # step_cnt = 0
        corr_cnt = 0
        running_f1 = 0.0
        running_macro_f1 = 0.0
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(test_loader):
                out = net(batch_x.to(hparam['device']))
                loss = loss_func(out, batch_y.to(hparam['device']))
                running_loss += loss.item()
                step_cnt += 1

                pred = compute_pred(out).flatten()
                batch_y = batch_y.data.cpu().numpy().flatten()
                accuracy = sum(pred == batch_y) / pred.shape[0]
                corr_cnt += accuracy

                macro_f1 = compute_macro_f1(batch_y, pred)
                running_macro_f1 += macro_f1

                f1 = compute_f1(batch_y, pred)
                running_f1 += f1
        test_acc = corr_cnt / len(test_loader)
        avg_test_loss = running_loss / len(test_loader)
        avg_test_macro_f1 = running_macro_f1 / len(test_loader)
        test_f1 = running_f1 / len(test_loader)
        print(' | Test Loss: {:.4f} | Acc: {:.4f} | F1: {:.4f}, {:.4f}'.format(avg_test_loss, test_acc, test_f1,
                                                                               avg_test_macro_f1))

        # Visualization
        if hparam['use_tensorboard']:
            writer.add_scalars('Training Loss Graph', {'train_loss': avg_train_loss,
                                                       'validation_loss': avg_valid_loss,
                                                       'test_loss': avg_test_loss}, epoch + 1)
            writer.add_scalars('Training Acc Graph', {'train_acc': train_acc,
                                                      'validation_acc': valid_acc,
                                                      'test_acc': test_acc}, epoch + 1)
            writer.add_scalars('Training Macro-F1 Graph', {'train_macro_f1': avg_train_macro_f1,
                                                           'validation_macro_f1': avg_valid_macro_f1,
                                                           'test_macro_f1': avg_test_macro_f1}, epoch + 1)
            writer.add_scalars('Training F1 Graph', {'train_f1': train_f1,
                                                     'validation_f1': valid_f1,
                                                     'test_f1': test_f1}, epoch + 1)



        time.sleep(0.5)

        if hparam['EARLY_STOP']:
            early_stopping(avg_valid_loss, net, epoch + 1)
            if early_stopping.early_stop == True:
                print("Early Stopping!")
                break
    with open(os.path.join(log_dir, 'best_epoch_{}.txt'.format(early_stopping.best_epoch)), 'w'):
        pass



def compute_pred(out):
    pred_y = out.detach()
    pred_y[pred_y >= 0.5] = 1
    pred_y[pred_y < 0.5] = 0
    pred_y = pred_y.data.cpu().numpy()
    return pred_y


def compute_macro_f1(target_y, pred_y):
    target_y = target_y.astype(np.int32)
    pred_y = pred_y.astype(np.int32)
    macro_f1 = f1_score(target_y, pred_y, average='macro')
    return macro_f1


def compute_f1(target_y, pred_y):
    target_y = target_y.astype(np.int32)
    pred_y = pred_y.astype(np.int32)
    f1 = f1_score(target_y, pred_y, average='binary')
    return f1


class MyDataset(Dataset):
    def __init__(self, data_path, split='train'):
        assert split in ['train', 'valid', 'test']

        self.data_root = data_path
        self.data_dir = jpath(self.data_root, 'data')
        self.meta = read_json(jpath(self.data_root, 'metadata.json'))
        self.fns = []

        # Put all sample path to self.fns
        for id in self.meta:
            if self.meta[id]['split'] == split:
                self.fns.append(jpath(self.data_root, self.meta[id]['path']))

    def __getitem__(self, idx):
        fn = self.fns[idx]
        column_names = ['timestamp', 'acc', 'acc0', 'acc1', 'acc2', 'gyro', 'gyro0', 'gyro1', 'gyro2', 'label']
        df = pd.read_csv(fn, names=column_names, skiprows=[0])
        label = list(df['label'])  # (T(500Hz))
        label_50 = [sum([label[i + j] for j in range(10)]) / 10 for i in range(0, len(label), 10)]
        label = torch.tensor(label_50, dtype=torch.float32)

        df = df.drop(['timestamp', 'label'], axis=1)
        signal = torch.tensor(df.values, dtype=torch.float32).T  # (C, T)

        return signal, label

    def __len__(self):
        return len(self.fns)





def check_model(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Totalparams:', format(pytorch_total_params, ','))
    print('Trainableparams:', format(pytorch_train_params, ','))
    return pytorch_train_params


if __name__ == '__main__':
	if len(sys.argv) == 2:
		hparam['dataset_root'] = sys.argv[1]
	main()
