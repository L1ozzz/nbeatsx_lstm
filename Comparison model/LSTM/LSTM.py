import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 读取数据
data = pd.read_csv('data/imputed_data_KNN_1.csv')

# 2. 数据预处理
data['Day(Local_Date)'] = pd.to_datetime(data['Day(Local_Date)'])
data = data.drop(['Day(Local_Date)', 'Season'], axis=1)

target_column = 'SoilM(%)'
features = data.drop([target_column], axis=1)
target = data[target_column]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

features_scaled = scaler_X.fit_transform(features)
target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

# 3. 构造时间序列数据：用过去 7 天的特征预测第 8 天的土壤湿度
sequence_length = 7
X, y = [], []
for i in range(len(features_scaled) - sequence_length):
    X.append(features_scaled[i : i + sequence_length])
    y.append(target_scaled[i + sequence_length])
X = np.array(X)  # shape: (样本数, 7, num_features)
y = np.array(y)  # shape: (样本数, 1)

# 4. 划分训练集和测试集（80% 训练，20% 测试）
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 为后续预测“最后 28 天”的测试效果，我们从测试集中取最后 28 个样本
X_test_last28 = X_test[-28:]
y_test_last28 = y_test[-28:]

# 5. 构建双层 LSTM 模型
model = Sequential()
# 第一层 LSTM：设置 return_sequences=True 以便后续堆叠第二层 LSTM
model.add(LSTM(256, activation='tanh', return_sequences=True, input_shape=(sequence_length, X.shape[2])))
model.add(Dropout(0.2))
# 第二层 LSTM：默认返回最后一个时间步的输出（return_sequences=False）
model.add(LSTM(256, activation='tanh'))
model.add(Dropout(0.2))
# 输出层
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.summary()

# 6. 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)

# 7. 预测最后28天的土壤湿度（递归预测）
predictions = []
# 取最后7天数据作为初始输入
last_7_days = features_scaled[-7:].reshape(1, sequence_length, X.shape[2])
for i in range(28):
    pred = model.predict(last_7_days)
    predictions.append(pred)
    # 使用 np.roll 进行数据位移，然后将最新预测值更新到最后一行（这里只更新目标列的位置）
    last_7_days = np.roll(last_7_days, shift=-1, axis=1)
    last_7_days[0, -1, 0] = pred

predictions_scaled = model.predict(X_test_last28)  # 每个样本窗口对应预测当天的土壤湿度

# 反归一化预测结果和真实值
predictions = scaler_y.inverse_transform(predictions_scaled)
y_test_inverse = scaler_y.inverse_transform(y_test_last28)

# 8. 输出预测值与真实值
print("预测值与真实值对比：")
for i in range(28):
    print(f"预测值: {predictions[i][0]:.2f}, 真实值: {y_test_inverse[i][0]:.2f}")

# 9. 绘制对比图
plt.plot(range(28), predictions, label='预测值', marker='o')
plt.plot(range(28), y_test_inverse, label='真实值', marker='o')
plt.xlabel('天数')
plt.ylabel('土壤湿度')
plt.title('预测值与真实值对比')
plt.legend()
plt.show()

# 10. 计算误差指标
mae_val = mean_absolute_error(y_test_inverse, predictions)
mse_val = mean_squared_error(y_test_inverse, predictions)
rmse_val = np.sqrt(mse_val)
smape_val = 100 * np.mean(2 * np.abs(predictions - y_test_inverse) / (np.abs(predictions) + np.abs(y_test_inverse)))
r2_val = 1 - np.sum((y_test_inverse - predictions)**2) / np.sum((y_test_inverse - np.mean(y_test_inverse))**2)

print(f"MAE: {mae_val:.2f}")
print(f"MSE: {mse_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")
print(f"sMAPE: {smape_val:.2f}%")
print(f"R^2: {r2_val:.4f}")

# 11. 按指定格式输出预测值和真实值
predictions_list = [round(val, 6) for val in predictions.flatten()]
y_test_array = y_test_inverse.flatten()
print(predictions_list)
print(y_test_array)
