import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 导入 TCN 层（需安装 keras-tcn）
from tcn import TCN

# 1. 读取数据
data = pd.read_csv('data/imputed_data_KNN_1.csv')

# 2. 数据预处理
# 将日期转换为 datetime 类型（后续未使用，可根据需要保留或删除）
data['Day(Local_Date)'] = pd.to_datetime(data['Day(Local_Date)'])
# 删除日期和季节列
data = data.drop(['Day(Local_Date)', 'Season'], axis=1)

# 目标列：土壤湿度
target_column = 'SoilM(%)'
features = data.drop([target_column], axis=1)
target = data[target_column]

# 归一化（这里使用 MinMaxScaler）
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

num_features = X.shape[2]

# 4. 划分训练集和测试集（80% 训练，20% 测试）
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 为后续预测“最后 28 天”的测试效果，我们从测试集中取最后 28 个样本
# 注意：由于每个样本对应一天的预测，所以最后28个样本即为最近28天
X_test_last28 = X_test[-28:]
y_test_last28 = y_test[-28:]

# 5. 构建两层 TCN 模型
# 第一层 TCN 设置 return_sequences=True，第二层 TCN 设置 return_sequences=False
model = Sequential([
    # 第一层 TCN：返回完整时序输出以供后续堆叠
    TCN(input_shape=(sequence_length, num_features),
        dilations=[1, 2, 4, 8, 16],
        nb_filters=32,
        kernel_size=3,
        dropout_rate=0.2,
        return_sequences=True),
    Dropout(0.2),
    # 第二层 TCN：只输出最后一个时间步的结果
    TCN(dilations=[1, 2, 4, 8, 16],
        nb_filters=32,
        kernel_size=3,
        dropout_rate=0.2,
        return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.summary()

# 6. 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)

# 7. 使用测试集中最后 28 个样本进行预测
predictions_scaled = model.predict(X_test_last28)  # shape: (28, 1)

# 反归一化：将预测结果和真实值转换回原始尺度
predictions = scaler_y.inverse_transform(predictions_scaled)
y_test_inverse = scaler_y.inverse_transform(y_test_last28)

# 8. 输出预测值与真实值，要求格式如下：
# 预测值格式为列表，保留6位小数
predictions_list = [round(val, 6) for val in predictions.flatten()]
# 真实值格式直接打印 numpy 数组（默认以空格分隔）
y_test_array = y_test_inverse.flatten()

print("预测值:")
print(predictions_list)
print("真实值:")
print(y_test_array)

# 9. 计算误差指标
mae_val = mean_absolute_error(y_test_inverse, predictions)
mse_val = mean_squared_error(y_test_inverse, predictions)
rmse_val = np.sqrt(mse_val)
smape_val = 100 * np.mean(2 * np.abs(predictions - y_test_inverse) / (np.abs(predictions) + np.abs(y_test_inverse)))
r2_val = r2_score(y_test_inverse, predictions)

print(f"MAE: {mae_val:.2f}")
print(f"MSE: {mse_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")
print(f"sMAPE: {smape_val:.2f}%")
print(f"R^2: {r2_val:.4f}")

# 10. 绘制预测值与真实值对比图（最后28天）
plt.figure(figsize=(10, 5))
plt.plot(range(28), predictions, label='预测值', marker='o')
plt.plot(range(28), y_test_inverse, label='真实值', marker='o')
plt.xlabel('天数')
plt.ylabel('土壤湿度')
plt.title('最后28天预测值与真实值对比')
plt.legend()
plt.grid(True)
plt.show()
