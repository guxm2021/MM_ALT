import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, savefolder=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.savefolder = savefolder
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.savefolder == None:
            torch.save(model.state_dict(), 'checkpoint.pth')
        else:
            torch.save(model.state_dict(), os.path.join(self.savefolder, 'checkpoint.pth'))

        self.val_loss_min = val_loss

import os
import json
from datetime import timedelta

jpath = os.path.join
ls = os.listdir



def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
    data = json.loads(data)
    return data


def save_json(data, path):
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


def print_json(data):
    '''
    Format print a json string
    '''
    print(json.dumps(data, indent=4, ensure_ascii=False))


def timecode_to_timedelta(timecode):
    '''
    Convert timecode 'MM:SS.XXX' to timedelta object
    '''
    m, s = timecode.strip().split(':')
    m = int(m)
    s = float(s)
    ret = timedelta(minutes=m, seconds=s)
    return ret


def sec_to_timedelta(time_in_sec):
    '''
    Convert 'sec.milli' to timedelta object
    :param time_in_sec: string, time in second
    '''
    time_in_sec = float(time_in_sec)
    ret = timedelta(seconds=time_in_sec)
    return ret


def main():
    pass


if __name__ == '__main__':
    main()
