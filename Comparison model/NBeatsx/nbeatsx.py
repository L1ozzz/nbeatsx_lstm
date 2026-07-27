import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from epfnbeatsx import EPF
from ts_dataset_nbeatsx import TimeSeriesDataset
from ts_loader_nbeatsx import TimeSeriesLoader
from nbeats import Nbeats
from keras.losses import mean_squared_error, mean_absolute_error

# 加载本地数据
Y_df, X_df, S_df = EPF.load(directory='./data')
Y_df.head()

'''
# 选择数值型列进行标准化
numerical_cols = X_df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_df[numerical_cols] = scaler.fit_transform(X_df[numerical_cols])

'''
# train_mask: 1 to keep, 0 to mask
train_mask = np.ones(len(Y_df))
train_mask[-3587:] = 0 # Last week of data (168 hours)

# 数据集对象。将 DataFrame 预处理为 pytorch 张量和窗口。
ts_dataset = TimeSeriesDataset(Y_df=Y_df, X_df=X_df, S_df=S_df, ts_train_mask=train_mask)
# Handling numerical and string data dynamically
numerical_cols = X_df.select_dtypes(include=[np.number]).columns
string_cols = X_df.select_dtypes(include=[object, 'category']).columns
label_encoders = {}
for col in string_cols:
    le = LabelEncoder()
    X_df[col] = le.fit_transform(X_df[col].astype(str))
    label_encoders[col] = le

# 打印 t_cols 确认
print(ts_dataset.t_cols)
# 加载对象。采样数据集对象的窗口。
# 有关每个参数的更多信息，请参阅 Loader 对象上的注释。
train_loader = TimeSeriesLoader(model='nbeats',
                                ts_dataset=ts_dataset,
                                window_sampling_limit=8969,  # 4 years of data
                                offset=0,
                                input_size=7,  # Last 7 days
                                output_size=1,  # Predict 1 day
                                idx_to_sample_freq=1,  # Sampling frequency of 1 day
                                batch_size=1024,
                                is_train_loader=True,
                                shuffle=True)

# 验证加载器（注意：在此示例中，我们还对预测期进行了验证）
val_loader = TimeSeriesLoader(model='nbeats',
                              ts_dataset=ts_dataset,
                              window_sampling_limit=8969,  # 4 years of data
                              offset=0,
                              input_size=7,  # Last 7 days
                              output_size=1,  # Predict 1 day
                              idx_to_sample_freq=1,  # Sampling frequency of 1 day
                              batch_size=1024,
                              is_train_loader=False,
                              shuffle=False)

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
               #initialization='glorot_normal',
               #activation='selu',
               initialization='he_normal',
               activation='relu',
               stack_types=['trend'] +['exogenous_tcn'], #['trend'] +['seasonality']  + ['identity'] + ['exogenous_tcn'], #+ ['exogenous_wavenet'],
               n_blocks=[2, 2],
               n_layers=[2, 2],
               n_hidden=[[512, 512], [512, 512]],
               n_harmonics=0,  # not used with exogenous_tcn
               n_polynomials=0,  # not used with exogenous_tcn
               x_s_n_hidden=0,
               exogenous_n_channels=9,
               include_var_dict=include_var_dict,
               t_cols=ts_dataset.t_cols,
               batch_normalization=True,
               dropout_prob_theta=0.2,
               dropout_prob_exogenous=0.2,
               learning_rate=0.0001,
               lr_decay=0.6,
               n_lr_decay_steps=5,
               early_stopping=10,
               weight_decay=0,
               l1_theta=0,
               n_iterations=5000,
               loss='MAE',
               loss_hypar=0.5,
               val_loss='MAE',
               seasonality=7,  # not used: only used with MASE loss
               random_seed=1)

model.fit(train_ts_loader=train_loader, val_ts_loader=val_loader, eval_steps=100)

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
        input_size=7,
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
predictions_tcn = []
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
    current_loader = create_loader(current_history, model='nbeats', input_size=30,history_data_x=history_data_x)
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
    predictions_tcn.append(y_hat_today)
    print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
import matplotlib.pyplot as plt

# 确保传递正确的数据列给绘图函数
plt.plot(range(history_length), history_data_y['y'], label='Historical Soil Moisture')

# 绘制预测值和实际值
x_values = range(history_length, history_length + forecast_length)
plt.plot(x_values, predictions_tcn, linestyle='dashed', label='Forecast', marker='o')
plt.plot(x_values, actuals, label='Actual Soil Moisture', marker='x')

# 添加图例和标签
plt.legend()
plt.grid(True)
plt.xlabel('Day')
plt.ylabel('Soil Moisture')
plt.show()


print(predictions_tcn)
print(actuals)

# 计算误差指标
mse_tcn = mean_squared_error(actuals, predictions_tcn)
mae_tcn = mean_absolute_error(actuals, predictions_tcn)
mape_tcn = np.mean(np.abs((actuals - np.array(predictions_tcn)) / actuals)) * 100
rmse_tcn = np.sqrt(mse_tcn)

print(f'MSE: {mse_tcn:.4f}')
print(f'MAE: {mae_tcn:.4f}')
print(f'MAPE: {mape_tcn:.2f}%')
print(f'RMSE: {rmse_tcn:.2f}%')

def calculate_smape(actuals, predictions):
    return np.mean(2 * np.abs(actuals - predictions) / (np.abs(actuals) + np.abs(predictions))) * 100

# 计算 sMAPE
smape_tcn = calculate_smape(actuals, predictions_tcn)

# 输出 sMAPE
print(f"sMAPE: {smape_tcn:.2f}%")

# 计算 R^2 决定系数（方法一：手动计算）
r2_tcn = 1 - np.sum((actuals - np.array(predictions_tcn))**2) / np.sum((actuals - np.mean(actuals))**2)
print(f'R^2 (manual): {r2_tcn:.4f}')

# 或者方法二：使用 sklearn
from sklearn.metrics import r2_score
r2_tcn = r2_score(actuals, predictions_tcn)
print(f'R^2 (sklearn): {r2_tcn:.4f}')