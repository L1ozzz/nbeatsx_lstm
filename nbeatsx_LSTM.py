from datetime import time

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from keras.losses import mean_squared_error, mean_absolute_error
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from nbeatsx_lstm_epf import EPF
from nbeatsx_lstm_dataset import TimeSeriesDataset
from nbeatsx_lstm_loader import TimeSeriesLoader
from nbeats_LSTM import Nbeats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn.functional as F

# 加载本地数据
directory='./data'
filename='imputed_data_KNN_1.csv'
Y_df, X_df, S_df = EPF.load(directory,filename)
#print(S_df)
#print(Y_df.head)
# 初始历史数据，包含最后30天的数据用于第一次预测




'''
# 选择数值型列进行标准化
numerical_cols = X_df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_df[numerical_cols] = scaler.fit_transform(X_df[numerical_cols])

'''
# train_mask: 1 to keep, 0 to mask
train_mask = np.ones(len(Y_df))
train_mask[-3587:] = 0 # Last week of data (168 hours)
#train_mask[-3600:] = 0

ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df, ts_train_mask=train_mask)


# Handling numerical and string data dynamically
numerical_cols = X_df.select_dtypes(include=[np.number]).columns
string_cols = X_df.select_dtypes(include=[object, 'category']).columns
label_encoders = {}
for col in string_cols:#处理非数值列
    le = LabelEncoder()
    X_df[col] = le.fit_transform(X_df[col].astype(str))
    label_encoders[col] = le

'''
his_data_x = Y_df.iloc[-(8 + 7):-7]
new_ = his_data_x.iloc[-1].copy()  # 复制最后一行以保持结构
print(new_[0])
print(new_['ds'])
'''
#new_entry['ds'] = future_dates[i]  # 更新日期为未来日期
#new_entry['y'] = actuals[i]  # 使用实际值而非预测值
    # 转换 unique_id 为整数，递增后转换回字符串
#new_['unique_id'] = str(int(his_data_x['unique_id'].iloc[-1]) + 1)

# 打印 t_cols 确认
#print(ts_dataset.t_cols)
# 加载对象。采样数据集对象的窗口。
# 有关每个参数的更多信息，请参阅 Loader 对象上的注释。
train_loader = TimeSeriesLoader(model='nbeats',
                                ts_dataset=ts_dataset,
                                window_sampling_limit=5382,  # 4 years of data
                                offset=0,
                                input_size=7,  # Last 7 days
                                output_size=1,  # Predict 1 day
                                idx_to_sample_freq=1,  # Sampling frequency of 1 day
                                batch_size=1024,
                                is_train_loader=True,
                                shuffle=False)

# 验证加载器（注意：在此示例中，我们还对预测期进行了验证）

val_loader = TimeSeriesLoader(model='nbeats',
                              ts_dataset=ts_dataset,
                              window_sampling_limit=3550,  # 4 years of data
                              offset=0,
                              input_size=7,  # Last 7 days
                              output_size=1,  # Predict 1 day
                              idx_to_sample_freq=1,  # Sampling frequency of 1 day
                              batch_size=1024,#1024
                              is_train_loader=False,
                              shuffle=False)

print(dir(val_loader))
# 包含要包含的滞后变量的字典。
include_var_dict = { 'y': list(range(-7, 0)),  # 过去30天的土壤湿度
                    'GustDir': list(range(-7, 0)),
                    'GustSpd': list(range(-7, 0)),
                    'WindRun': list(range(-7, 0)),
                    'Rain': list(range(-7, 0)),
                    'Tmean': list(range(-7, 0)),
                    'Tmax': list(range(-7, 0)),
                    'Tmin': list(range(-7, 0)),
                    'Tgmin': list(range(-7, 0)),
                    'VapPress': list(range(-7, 0)),
                    'ET10': list(range(-7, 0)),
                    'Rad': list(range(-7, 0)),
                    'week_day': [-1]}  # Last day of the week

model = Nbeats(input_size_multiplier=7,  # Last 7 days
               output_size=1,  # Predict 1 day
               shared_weights=False,
               initialization='he_normal',#'he_uniform'
               activation='relu',
               stack_types=['trend']+['exogenous_lstm']+ ['seasonality'],
               n_blocks=[2, 2, 2],
               n_layers=[2, 2, 2],
               n_hidden=[[512, 512], [512, 512], [512, 512]],
               #n_blocks=[4, 4, 4, 4],
               #n_layers=[4, 4, 4, 4],
               #n_hidden=[[1024, 1024,1024, 1024], [1024, 1024,1024, 1024],[512,512,512,512], [1024,1024,1024,1024]],
               #stack_types=['seasonality']+['identity']+ ['exogenous_lstm'],
               #n_blocks=[4, 4, 4], n_layers=[4, 4, 16],
               #n_hidden=[[1024, 1024, 1024, 1024], [1024, 1024, 1024, 1024],  [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]],
               # stack_types=['seasonality']+['identity']+ ['exogenous_lstm']+['trend'] ,
               # n_blocks=[2, 2, 2,2],
               # n_layers=[2, 2, 2,2],
               # n_hidden=[[512, 512], [512, 512], [512, 512], [512, 512]],
               n_harmonics=1,  # not used with exogenous_tcn
               n_polynomials=4,  # not used with exogenous_tcn
               x_s_n_hidden=0,
               exogenous_n_channels=len(string_cols),
               include_var_dict=include_var_dict,
               t_cols=ts_dataset.t_cols,
               batch_normalization=True,
               dropout_prob_theta=0.3,#0.5
               dropout_prob_exogenous=0.3,#0.5
               learning_rate=0.0002,#0.000977993
               lr_decay=0.9,#0.9
               n_lr_decay_steps=50,#3
               early_stopping=10,#10
               weight_decay=0.1,#0.00500772
               l1_theta=0,
               n_iterations=5000,
               loss='MAE',
               loss_hypar=0.5,
               val_loss='MAE',
               seasonality=7,  # not used: only used with MASE loss
               random_seed=1)


model.fit(train_ts_loader=train_loader, val_ts_loader=val_loader, eval_steps=50)
# 保存模型的状态字典
model_dir = 'model'  # 指定保存模型的路径
model_id = 'nbeats_lstm_model'  # 给模型命名
print(time)
# 保存模型
model.save(model_dir, model_id)
print(val_loader)

import numpy as np
import pandas as pd


history_length = 30  # 用于绘制的历史天数
forecast_length = 28  # 预测天数
#history_data_x = X_df.iloc[-(history_length + forecast_length):-forecast_length]
# 初始历史数据，包含最后30天的数据用于第一次预测
history_data_y = Y_df.iloc[-(history_length + forecast_length):-forecast_length]
# 数据集对象。将 DataFrame 预处理为 pytorch 张量和窗口。

test_mask = np.ones(len(history_data_y))
test_mask[-1:] = 0

def create_loader(current_history, model, input_size,history_data_x, batch_size=1):
    # 将current_history转换为DataFrame，因为TimeSeriesDataset可能需要DataFrame输入
    print("current_history: ", current_history)
    #test_mask = np.ones(len(current_history))
    #test_mask[-30:] = 0
    data_df = pd.DataFrame(current_history)
    datax_df = pd.DataFrame(history_data_x)
    print("data_df: ",data_df)
    test_dataset = TimeSeriesDataset(Y_df=data_df, X_df=datax_df, S_df=S_df, ts_train_mask=test_mask)
    test_loader = TimeSeriesLoader(
        model='nbeats',
        ts_dataset=test_dataset,
        window_sampling_limit=8,
        offset=0,
        input_size=input_size,
        output_size=1,
        idx_to_sample_freq=1,
        batch_size=1,
        is_train_loader=False,
        shuffle=False
        )
    return test_loader



# 设定历史数据和预测数据的长度

import numpy as np

# 初始化存储预测和实际结果的列表
predictions = []
actuals = Y_df['y'].values[-forecast_length:]  # 实际值
print(actuals)
last_known_date = history_data_y['ds'].iloc[-1]
future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=forecast_length, freq='D')
print("日期们： ",future_dates)

# 进行递归预测
current_history = history_data_y.copy()
for i in range(forecast_length):
    print("this is iteration:",i)

    # 动态生成history_data_x
    start_idx = -(history_length + forecast_length) + i
    end_idx = -forecast_length + i if -forecast_length + i != 0 else None
    history_data_x = X_df.iloc[start_idx:end_idx]

    start_idx_1 = -(history_length + forecast_length) + i
    end_idx_1 = -forecast_length + i if -forecast_length + i != 0 else None
    current_history = Y_df.iloc[start_idx_1:end_idx_1]
    #print(current_history)

    # 创建当前历史数据的加载器
    current_loader = create_loader(current_history, model='nbeats', input_size=7,history_data_x=history_data_x)
    # 使用模型进行预测
    y_true,y_hat_today, *_ = model.predict(ts_loader=current_loader, return_decomposition=False) # 获取预测结果
    #y_hat = model.predict(ts_loader=current_loader, return_decomposition=False)
    #print("This is Y-hat:",y_hat)
    #print(y_hat_today)
    y_hat_today = y_hat_today.flatten()
    print("这是扁平：",y_hat_today)
    y_hat_today = y_hat_today[-1]# 假设每次预测输出一个结果
    print("迭代次数：",i, "预测结果：",y_hat_today)

    # 存储预测结果
    predictions.append(y_hat_today)
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
import matplotlib.pyplot as plt

# 确保传递正确的数据列给绘图函数
plt.plot(range(history_length), history_data_y['y'], label='Historical Soil Moisture')

# 绘制预测值和实际值
x_values = range(history_length, history_length + forecast_length)
plt.plot(x_values, predictions, linestyle='dashed', label='Forecast', marker='o')
plt.plot(x_values, actuals, label='Actual Soil Moisture', marker='x')

# 添加图例和标签
plt.legend()
plt.grid(True)
plt.xlabel('Day')
plt.ylabel('Soil Moisture')
plt.show()


print(predictions)
print(actuals)





# 假设 predictions 和 actuals 是 Python 列表
predictions_list = predictions  # 你的预测列表
actuals_list = actuals  # 你的实际值列表

# 将列表转换为 PyTorch 的 Tensor
predictions_tensor = torch.tensor(predictions_list, dtype=torch.float32)
actuals_tensor = torch.tensor(actuals_list, dtype=torch.float32)

# MSE (Mean Squared Error) 使用 torch.nn.functional.mse_loss
mse = F.mse_loss(predictions_tensor, actuals_tensor)

# RMSE (Root Mean Squared Error)
rmse = torch.sqrt(mse)

# MAE (Mean Absolute Error) 使用 torch.nn.functional.l1_loss
mae = F.l1_loss(predictions_tensor, actuals_tensor)

# MAPE (Mean Absolute Percentage Error)
mape = torch.mean(torch.abs((predictions_tensor - actuals_tensor) / actuals_tensor)) * 100

# 打印结果
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.2f}%')




import matplotlib.pyplot as plt

# 获取训练和验证损失
train_losses = model.trajectories['train_loss']
val_losses = model.trajectories['val_loss']
iterations = model.trajectories['iteration']

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(iterations, train_losses, label='Training Loss')
#plt.plot(iterations, val_losses, label='Validation Loss')
#plt.title('Training Loss')  # 去掉标题
plt.xlabel('Iteration', fontsize=14)  # 增大横坐标文字字号
plt.ylabel('Loss', fontsize=14)  # 增大纵坐标文字字号
# plt.legend()  # 去掉legend
plt.grid(True)
plt.savefig("train_loss.png")
plt.show()


# 绘制损失曲线
plt.figure(figsize=(10, 5))
#plt.plot(iterations, train_losses, label='Training Loss')
plt.plot(iterations, val_losses, label='Validation Loss')
#plt.title('Validation Loss', fontsize=14)
plt.xlabel('Iteration',fontsize=14)
plt.ylabel('Loss',fontsize=14)
#plt.legend()
plt.grid(True)
plt.savefig("val_loss.png")
plt.show()

smape = torch.mean(2 * torch.abs(predictions_tensor - actuals_tensor) / (torch.abs(predictions_tensor) + torch.abs(actuals_tensor))) * 100

# 打印结果
print(f'sMAPE: {smape:.2f}%')

