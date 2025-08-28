import os
from pathlib import Path
from datetime import timedelta
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 导入你定义的模块（确保这些模块在你的 PYTHONPATH 中）
from nbeatsx_lstm_epf import EPF
from nbeatsx_lstm_dataset import TimeSeriesDataset
from nbeatsx_lstm_loader import TimeSeriesLoader
from nbeats_LSTM import Nbeats

# ========================
# 1. 加载数据和预处理
# ========================

directory = './data'
filename = 'imputed_data_KNN_1.csv'
Y_df, X_df, S_df = EPF.load(directory, filename)

# 对非数值列进行标签编码（与训练时一致）
from sklearn.preprocessing import LabelEncoder

string_cols = X_df.select_dtypes(include=[object, 'category']).columns
label_encoders = {}
for col in string_cols:
    le = LabelEncoder()
    X_df[col] = le.fit_transform(X_df[col].astype(str))
    label_encoders[col] = le

# ========================
# 2. 定义测试参数
# ========================

history_length = 365  # 用于预测时的历史长度
forecast_length = 365  # 预测天数

# 获取历史数据（用于第一次预测的基础）
history_data_y = Y_df.iloc[-(history_length + forecast_length):-forecast_length]

# 构造一个 mask（根据你的需求，此处与训练时保持一致）
test_mask = np.ones(len(history_data_y))
test_mask[-1:] = 0

# ========================
# 3. 实例化模型并加载已保存的权重
# ========================

# 注意：下面的超参数需要与训练时使用的一致！
include_var_dict = {
    'y': list(range(-7, 0)),
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
    'week_day': [-1]
}

# 假设 ts_dataset 在训练时有一个属性 t_cols（例如时间列信息），这里需要保持一致
# 如果你有这个变量，可以直接传入，如：t_cols = ts_dataset.t_cols
# 这里示例直接设定为 None 或根据实际情况赋值
t_cols = None

# 构造模型对象
model = Nbeats(
    input_size_multiplier=7,
    output_size=1,
    shared_weights=False,
    initialization='he_normal',
    activation='relu',
    stack_types=['trend'] + ['exogenous_lstm'] + ['seasonality'],
    n_blocks=[2, 2, 2],
    n_layers=[2, 2, 2],
    n_hidden=[[512, 512], [512, 512], [512, 512]],
    n_harmonics=1,
    n_polynomials=4,
    x_s_n_hidden=0,
    exogenous_n_channels=len(string_cols),
    include_var_dict=include_var_dict,
    t_cols=t_cols,
    batch_normalization=True,
    dropout_prob_theta=0.3,
    dropout_prob_exogenous=0.3,
    learning_rate=0.0002,
    lr_decay=0.9,
    n_lr_decay_steps=50,
    early_stopping=10,
    weight_decay=0.1,
    l1_theta=0,
    n_iterations=5000,
    loss='MAE',
    loss_hypar=0.5,
    val_loss='MAE',
    seasonality=7,
    random_seed=1
)
'''
# 加载保存的模型权重


model.load(model_dir, model_id)
print("模型加载成功！")
'''
model_dir = 'model'
model_id = 'nbeats_lstm_model'
model_file = os.path.join(model_dir, f"model_{model_id}.pth")
assert os.path.isfile(model_file), "No model file found at {}".format(model_file)
print('Loading entire model from:\n {}'.format(model_file) + '\n')
model = torch.load(model_file, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# ========================
# 4. 定义一个函数，用于生成当前预测所需的 Loader
# ========================

def create_loader(current_history, input_size, history_data_x, batch_size=1):
    """
    参数:
        current_history: 当前用于预测的目标变量历史数据（DataFrame 格式）
        input_size: 模型输入长度（例如7天）
        history_data_x: 与目标变量对应的解释变量（DataFrame 格式）
        batch_size: 批量大小（预测时通常设为1）
    """
    data_df = pd.DataFrame(current_history)
    datax_df = pd.DataFrame(history_data_x)
    test_dataset = TimeSeriesDataset(Y_df=data_df, X_df=datax_df, S_df=S_df, ts_train_mask=test_mask)
    test_loader = TimeSeriesLoader(
        model='nbeats',
        ts_dataset=test_dataset,
        window_sampling_limit=8,  # 这个参数保持和训练时一致或根据需要调整
        offset=0,
        input_size=input_size,
        output_size=1,
        idx_to_sample_freq=1,
        batch_size=batch_size,
        is_train_loader=False,
        shuffle=False
    )
    return test_loader


# ========================
# 5. 利用加载的模型进行递归预测
# ========================

predictions = []
actuals = Y_df['y'].values[-forecast_length:]  # 真实值
last_known_date = history_data_y['ds'].iloc[-1]
future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1),
                             periods=forecast_length, freq='D')
print("预测日期：", future_dates)

trend_preds, seasonal_preds, exo_preds = [], [], []

# n_blocks 对应 ['trend','exogenous_lstm','seasonality']
n_blocks = [2, 2, 2]

# 递归预测
current_history = history_data_y.copy()
for i in range(forecast_length):
    #print(f"第 {i} 次迭代预测...")
    # 根据当前迭代的索引动态更新输入数据
    start_idx = -(history_length + forecast_length) + i
    end_idx = -forecast_length + i if (-forecast_length + i) != 0 else None
    history_data_x = X_df.iloc[start_idx:end_idx]
    current_history = Y_df.iloc[start_idx:end_idx]

    # 创建当前时刻的 Loader
    current_loader = create_loader(current_history, input_size=7, history_data_x=history_data_x)

    # 调用模型的 predict 函数，获得当前时刻的预测值
    # 注意：这里的 predict 函数返回值需要和你的模型实现保持一致
    # ① 调用 predict 时，加 return_decomposition=True
    #y_true, y_hat_today, decomposition = model.predict(
    #    ts_loader=current_loader,
    #    return_decomposition=True
    #)
    # 3) 调用 predict，接收所有输出
    #outputs = model.predict(ts_loader=current_loader,
    #                        return_decomposition=True)

    # outputs[0] = y_true， outputs[1] = y_hat array，
    # outputs[2] = block-level contributions array，
    # outputs[3] 可能是其它返回值
    #y_hat_arr = outputs[1]  # shape (batch, 1)
    #y_hat_today = float(y_hat_arr.flatten()[-1])  # 标量预测值
    y_true, y_hat_today, *_ = model.predict(ts_loader=current_loader, return_decomposition=False)
    y_hat_today = y_hat_today.flatten()[-1]  # 获取当前时刻的最后一个预测值


    # ② 从 decomposition 中提取每个堆栈的贡献
    #    注意：key 名称要与模型内部保持一致
    #    有的实现叫 'seasonal' 或 'seasonality'，请根据你的源码确认
    #trend_val = decomposition['trend'].flatten()[-1]
    #seasonal_val = decomposition['seasonal'].flatten()[-1]
    #exo_val = decomposition['exogenous_lstm'].flatten()[-1]
    #print(f"迭代 {i} 的预测值：{y_hat_today}")
    predictions.append(y_hat_today)
    '''
    # outputs = model.predict(..., return_decomposition=True)
    # outputs[2] 是 backcast，outputs[3] 才是 forecast decomposition
    # outputs = model.predict(..., return_decomposition=True)
    fc = outputs[3]
    print("fc.shape:", fc.shape)       # 看看到底是 (6,1) 还是 (1,6)
    block_fc = fc.flatten()
    print("block_fc:", block_fc[:6])   # 核实一下前几项


    # 按 n_blocks=[2,2,2] 切片求和
    n_blocks = [2,2,2]
    start = 0
    trend_val    = block_fc[start:start+n_blocks[0]].sum()
    start += n_blocks[0]
    exo_val      = block_fc[start:start+n_blocks[1]].sum()
    start += n_blocks[1]
    seasonal_val = block_fc[start:start+n_blocks[2]].sum()

    trend_preds.append(trend_val)
    exo_preds.append(exo_val)
    seasonal_preds.append(seasonal_val)
    '''

    # ③ 存入各自列表
    #trend_preds.append(trend_val)
    #seasonal_preds.append(seasonal_val)
    #exo_preds.append(exo_val)

# ========================
# 6. 绘制预测结果与实际值对比
# ========================
'''
plt.figure(figsize=(10, 5))
# 绘制历史数据
plt.plot(range(history_length), history_data_y['y'], label='Historical Soil Moisture')
# 绘制预测值与实际值（x轴坐标从 history_length 开始）
x_values = range(history_length, history_length + forecast_length)
plt.plot(x_values, predictions, linestyle='dashed', marker='o', label='Forecast')
plt.plot(x_values, actuals, marker='x', label='Actual Soil Moisture')
plt.xlabel('Day')
plt.ylabel('Soil Moisture')
plt.title('Forecast vs Actual')
plt.legend()
plt.grid(True)
plt.show()
'''
historical_to_plot = history_data_y['y'].iloc[-7:]
plt.figure(figsize=(10, 5))
plt.plot(range(7), historical_to_plot, label='Historical Soil Moisture')

# 绘制预测值与实际值（x轴坐标从7开始）
x_values = range(7, 7 + forecast_length)
plt.plot(x_values, predictions, linestyle='dashed', marker='o', label='Forecast')
plt.plot(x_values, actuals, marker='x', label='Actual Soil Moisture')
plt.xlabel('Day')
plt.ylabel('Soil Moisture')
plt.title('Forecast vs Actual')
plt.legend()
plt.grid(True)
plt.show()
# ========================
# 7. 计算误差指标
# ========================

predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
actuals_tensor = torch.tensor(actuals, dtype=torch.float32)

mse = F.mse_loss(predictions_tensor, actuals_tensor)
rmse = torch.sqrt(mse)
mae = F.l1_loss(predictions_tensor, actuals_tensor)
mape = torch.mean(torch.abs((predictions_tensor - actuals_tensor) / actuals_tensor)) * 100
smape = torch.mean(2 * torch.abs(predictions_tensor - actuals_tensor) /
                   (torch.abs(predictions_tensor) + torch.abs(actuals_tensor))) * 100

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.2f}%')
print(f'sMAPE: {smape:.2f}%')

# ========================
# 8. 绘制残差分布及自相关图
# ========================

residuals = predictions_tensor - actuals_tensor

plt.figure(figsize=(10, 5))
plt.hist(residuals.detach().cpu().numpy(), bins=30, color='red', alpha=0.7)
#plt.title('Error Distribution')
plt.xlabel('Prediction Error',fontsize =14)
plt.ylabel('Frequency',fontsize =14)
plt.grid(True)
plt.show()

# 残差时序图
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(residuals)), residuals.detach().cpu().numpy(), marker='o', linestyle='--')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals over Time')
plt.xlabel('Forecast Horizon (day index)')
plt.ylabel('Residual (Predicted - Actual)')
plt.grid(True)
plt.show()

# 如果需要进一步绘制自相关图（需安装 statsmodels）
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(10, 4))
plot_acf(residuals.detach().cpu().numpy(), lags=6)
plt.title('Autocorrelation of Residuals')
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(residuals.detach().cpu().numpy(), lags=6)
plt.title('Partial Autocorrelation of Residuals')
plt.show()

# ========================
# 9. 计算 R^2
# ========================

actual_mean = torch.mean(actuals_tensor)
ss_res = torch.sum((actuals_tensor - predictions_tensor) ** 2)
ss_tot = torch.sum((actuals_tensor - actual_mean) ** 2)
r2 = 1 - ss_res / ss_tot
print(f'R^2: {r2:.4f}')

# ========================
# 4.1 时间序列分解及分解后模型评估
# ========================

# 假设你已有 forecast_length 和历史窗口设置等

# ========================
# 4.1 时间序列分解及分解后模型评估
# ========================

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ① 将预测值和实际值构造成带有日期索引的 pandas Series
#    这里我们使用之前生成的 future_dates（预测日期）作为索引
forecast_series = pd.Series(np.array(predictions).flatten(), index=future_dates)
actual_series   = pd.Series(np.array(actuals).flatten(), index=future_dates)

# ② 进行加法分解
#    假设季节性周期为 7 天（可根据数据特性调整）
decomposition_actual   = seasonal_decompose(actual_series, model='additive', period=7)
decomposition_forecast = seasonal_decompose(forecast_series, model='additive', period=7)

# ③ 提取分量
trend_actual    = decomposition_actual.trend
seasonal_actual = decomposition_actual.seasonal
resid_actual    = decomposition_actual.resid

trend_forecast    = decomposition_forecast.trend
seasonal_forecast = decomposition_forecast.seasonal
resid_forecast    = decomposition_forecast.resid
'''
# ④ 绘制实际值分解结果
plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(actual_series, label='Original Actual')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend_actual, label='Trend Actual', color='orange')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal_actual, label='Seasonal Actual', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(resid_actual, label='Residual Actual', color='red')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# ⑤ 绘制预测值分解结果
plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(forecast_series, label='Original Forecast')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend_forecast, label='Trend Forecast', color='orange')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal_forecast, label='Seasonal Forecast', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(resid_forecast, label='Residual Forecast', color='red')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
'''
# ⑥ 分解后模型评估
#     为了评估趋势、季节性和残差部分的预测效果，这里以均方误差（MSE）为例
#     注意：分解过程中会产生 NaN（主要出现在边缘部分），因此我们在计算指标时先剔除这些空值

# 趋势部分
trend_actual_drop    = trend_actual.dropna()
trend_forecast_drop  = trend_forecast.dropna()
trend_mse = mean_squared_error(trend_actual_drop, trend_forecast_drop)

# 季节性部分
seasonal_actual_drop   = seasonal_actual.dropna()
seasonal_forecast_drop = seasonal_forecast.dropna()
seasonal_mse = mean_squared_error(seasonal_actual_drop, seasonal_forecast_drop)

# 残差部分
resid_actual_drop   = resid_actual.dropna()
resid_forecast_drop = resid_forecast.dropna()
resid_mse = mean_squared_error(resid_actual_drop, resid_forecast_drop)

print(f"Trend MSE: {trend_mse:.4f}")
print(f"Seasonal MSE: {seasonal_mse:.4f}")
print(f"Residual MSE: {resid_mse:.4f}")

# 如果需要，也可以计算 MAE 或其他指标：
trend_mae = mean_absolute_error(trend_actual_drop, trend_forecast_drop)
seasonal_mae = mean_absolute_error(seasonal_actual_drop, seasonal_forecast_drop)
resid_mae = mean_absolute_error(resid_actual_drop, resid_forecast_drop)

print(f"Trend MAE: {trend_mae:.4f}")
print(f"Seasonal MAE: {seasonal_mae:.4f}")
print(f"Residual MAE: {resid_mae:.4f}")

from statsmodels.tsa.seasonal import STL

# 确保使用反归一化后的原始数据
forecast_series = pd.Series(np.array(predictions).flatten(), index=future_dates)
actual_series   = pd.Series(np.array(actuals).flatten(), index=future_dates)

# 使用 STL 分解
stl_actual = STL(actual_series, period=7)
res_actual = stl_actual.fit()
trend_actual = res_actual.trend
seasonal_actual = res_actual.seasonal
resid_actual = res_actual.resid

stl_forecast = STL(forecast_series, period=7)
res_forecast = stl_forecast.fit()
trend_forecast = res_forecast.trend
seasonal_forecast = res_forecast.seasonal
resid_forecast = res_forecast.resid

# 绘制分解结果
plt.figure(figsize=(12, 8))

# 实际值分解结果绘制
plt.subplot(411)
plt.plot(actual_series, label='Original Actual', color='blue')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend_actual, label='Trend Actual', color='orange')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal_actual, label='Seasonal Actual', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(resid_actual, label='Residual Actual', color='red')
plt.legend(loc='upper left')
plt.tight_layout()
plt.suptitle("STL Decomposition of Actual Series", y=1.02)
plt.show()

# 添加预测值分解图的绘制
plt.figure(figsize=(12, 8))

# 预测值分解结果绘制
plt.subplot(411)
plt.plot(forecast_series, label='Original Forecast', color='blue')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend_forecast, label='Trend Forecast', color='orange')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal_forecast, label='Seasonal Forecast', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(resid_forecast, label='Residual Forecast', color='red')
plt.legend(loc='upper left')
plt.tight_layout()
plt.suptitle("STL Decomposition of Forecast Series", y=1.02)
plt.show()


# 计算误差指标（仅作为示例）
from sklearn.metrics import mean_squared_error, mean_absolute_error
trend_mse = mean_squared_error(trend_actual.dropna(), trend_forecast.dropna())
seasonal_mse = mean_squared_error(seasonal_actual.dropna(), seasonal_forecast.dropna())
resid_mse = mean_squared_error(resid_actual.dropna(), resid_forecast.dropna())

print(f"Trend MSE: {trend_mse:.4f}")
print(f"Seasonal MSE: {seasonal_mse:.4f}")
print(f"Residual MSE: {resid_mse:.4f}")



