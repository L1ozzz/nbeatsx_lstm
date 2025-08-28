import torch
import torch.nn as nn

# class LSTMBlock(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
#         super(LSTMBlock, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_length, hidden_size]
#
#         # 通过线性层，将每个时间步的输出映射到 output_size
#         lstm_out = self.fc(lstm_out)  # lstm_out: [batch_size, seq_length, output_size]
#         return lstm_out


class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
        super(LSTMBlock, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc_res=nn.Linear(input_size,hidden_size*2)
        self.fc_out = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm1(x)# lstm_out: [batch_size, seq_length, hidden_size]
        res_x=self.fc_res(x)
        conbined_out=torch.cat([lstm_out1,lstm_out2],dim=2)
        conbined_out=conbined_out+res_x

        # 通过线性层，将每个时间步的输出映射到 output_size
        lstm_out = self.fc_out(conbined_out)  # lstm_out: [batch_size, seq_length, output_size]
        return lstm_out




# class LSTMBlock(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_prob):
#         super(LSTMBlock, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
#         self.fc = nn.Linear(hidden_size, output_size)
#         # 使用卷积层代替线性层，提取局部特征
#         self.conv = nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
#         # Dropout层
#         self.dropout = nn.Dropout(dropout_prob)
#
#     def forward(self, x):
#         # x: [batch_size, seq_length, input_size]
#         x = x.permute(0, 2, 1)  # 将输入数据从 [batch_size, seq_length, input_size] 转换为 [batch_size, input_size, seq_length]
#
#         # 通过卷积层提取局部特征
#         x = self.conv(x)  # [batch_size, input_size, seq_length]
#
#         # 将数据回到 LSTM 输入的形状
#         x = x.permute(0, 2, 1)  # [batch_size, seq_length, input_size]
#
#         # 通过 LSTM 层
#         lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_length, hidden_size]
#
#         # 使用 Dropout 进行正则化
#         # lstm_out = self.dropout(lstm_out)
#
#         # 通过全连接层，将 LSTM 的输出映射到输出空间
#         lstm_out = self.fc(lstm_out)  # [batch_size, seq_length, output_size]
#
#         return lstm_out




