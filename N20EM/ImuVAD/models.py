import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    m = IMU_CRNN_GRU()
    check_model(m)

    # m = nn.LSTM(input_size=256, hidden_size=512, num_layers=2,
    #             bias=True, batch_first=True, dropout=0.0, bidirectional=True)
    x = torch.rand(size=(16, 8, 500))
    y = m(x)
    print(y.shape)


def check_model(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Totalparams:', format(pytorch_total_params, ','))
    print('Trainableparams:', format(pytorch_train_params, ','))


class IMU_CRNN_GRU(nn.Module):
    '''
    GRU 2 with fewer neurons
    ***Final choice***
    '''

    def __init__(self, dropout_cnn=0.2, dropout_rnn=0., rnn_width=60):
        super().__init__()

        channel_num_1 = 128
        channel_num_2 = 200

        self.down = nn.AvgPool1d(kernel_size=10, stride=5, padding=4)

        self.conv1 = nn.Conv1d(in_channels=8, out_channels=channel_num_1, kernel_size=3, stride=1,
                               padding=1)  # floor(500 + 2*p - 3 + 1) = 500
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.norm1 = nn.BatchNorm1d(num_features=channel_num_1)
        self.drop1 = nn.Dropout(p=dropout_cnn)

        self.conv2 = nn.Conv1d(in_channels=channel_num_1, out_channels=channel_num_2, kernel_size=3, stride=1,
                               padding=1)  # (250 + 2*2 - 4) / 1 = 250
        self.norm2 = nn.BatchNorm1d(num_features=channel_num_2)
        self.drop2 = nn.Dropout(p=dropout_cnn)  # [B, C2, T]

        self.rnn = nn.GRU(input_size=channel_num_2, hidden_size=rnn_width, num_layers=2,
                          bias=True, batch_first=True, dropout=dropout_rnn, bidirectional=True)
        self.drop3 = nn.Dropout(p=dropout_rnn)

        self.fc = nn.Linear(in_features=rnn_width * 2, out_features=1)

    def forward(self, x):
        if ('CUDA_VISIBLE_DEVICES' not in os.environ) or len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            self.rnn.flatten_parameters()
        x = self.down(x)  # [B, 64, 500]

        x = F.relu_(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.drop1(x)  # [B, C1=200, T=50]

        x = F.relu_(self.conv2(x))
        x = self.norm2(x)
        x = self.drop2(x)  # [B, C2=200, T=25]

        x = torch.permute(x, [0, 2, 1])  # [B, T=25, C2=256]
        x, _ = self.rnn(x)  # [B, T=25, 512]
        x = self.drop3(x)
        x = torch.sigmoid(self.fc(x))
        x = x.squeeze()

        return x


if __name__ == '__main__':
    main()
