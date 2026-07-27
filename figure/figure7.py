import matplotlib.pyplot as plt
import numpy as np

# Dates for x-axis
dates = ['2024-04-26', '2024-04-27', '2024-04-28', '2024-04-29', '2024-04-30',
         '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05',
         '2024-05-06', '2024-05-07', '2024-05-08', '2024-05-09', '2024-05-10',
         '2024-05-11', '2024-05-12', '2024-05-13', '2024-05-14', '2024-05-15',
         '2024-05-16', '2024-05-17', '2024-05-18', '2024-05-19', '2024-05-20',
         '2024-05-21', '2024-05-22', '2024-05-23']

# Prediction values for each model
nbeatsx = [44.07648, 41.870758, 41.801975, 41.034286, 40.73836, 41.16614, 40.102654, 40.79751, 42.22746, 42.347652,
           41.763226, 42.52207, 41.968987, 41.28875, 40.559925, 41.537342, 40.45698, 40.286976, 40.39652, 39.959896,
           42.751835, 41.05461, 42.53187, 43.935577, 44.963207, 45.262733, 47.63978, 48.481667]
actual = [41.8, 41.6, 41.4, 41.3, 40.2, 41.1, 42.7, 42.7, 42.6, 42.3, 42.2, 41.9, 41.7, 41.5, 41.2, 41.0, 40.8, 40.5,
          40.1, 41.2, 42.6, 43.2, 44.7, 44.5, 45.8, 50.4, 48.0, 48.5]
nbeatsx_lstm = [41.832764, 41.638435, 41.479164, 41.149536, 41.235874, 41.365166, 39.914238, 40.91834, 42.327972, 42.548023,
                42.34144, 42.09719, 41.823433, 41.47335, 41.247906, 40.965076, 40.76791, 40.580376, 40.51201, 40.202126,
                41.855244, 41.12413, 42.177654, 42.60577, 44.199615, 50.238243, 45.813404, 50.23999]
lstm = [41.171345, 41.227646, 37.075787, 37.463654, 38.111233, 41.01868, 41.214012, 42.25341, 43.249344, 43.52793,
        43.00155, 42.1674, 40.79034, 40.191944, 40.642906, 40.241898, 40.456566, 41.95825, 42.826656, 43.854183,
        48.074566, 48.27449, 47.12295, 45.871544, 44.919483, 49.155876, 48.79613, 47.784702]
tcn = [41.686676, 41.860634, 39.276432, 40.78327, 42.830025, 46.06664, 45.630302, 45.1243, 45.8483, 45.197533,
       43.308765, 45.73366, 43.08148, 41.92049, 41.938206, 42.990112, 41.469147, 43.093945, 43.65639, 42.861225,
       48.481724, 46.598606, 46.15352, 44.105724, 47.07728, 48.395878, 47.122963, 47.37285]

# Plotting
plt.figure(figsize=(10, 6))

# Plot each model's prediction values
plt.plot(dates, nbeatsx, 'o-', label='NBeatsx', color='green')
plt.plot(dates, actual, 's-', label='Actual', color='black')
plt.plot(dates, nbeatsx_lstm, '^-', label='NBeatsx-LSTM', color='orange')
plt.plot(dates, lstm, 'v-', label='LSTM', color='blue')
plt.plot(dates, tcn, 'D-', label='TCN', color='grey')

# Adding labels and title
plt.xlabel('Date', fontsize=16)
plt.ylabel('Prediction Value', fontsize=16)
#plt.title('Comparison of Prediction Models', fontsize=16)
plt.xticks(rotation=45, fontsize=12)  # Rotate x labels for better visibility
plt.legend(loc='upper left', fontsize=13)

# Add gridlines to the background
plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()

import itertools
from scipy.stats import ttest_rel
import seaborn as sns

# ---- 1. 把你的预测列表转换成 np.array ----
preds = {
    'NBeatsx':     np.array(nbeatsx),
    'NBeatsx-LSTM':np.array(nbeatsx_lstm),
    'LSTM':        np.array(lstm),
    'TCN':         np.array(tcn),
}
names = list(preds.keys())

# ---- 2. 计算每个模型的绝对误差序列 ----
errors = { name: np.abs(preds[name] - np.array(actual)) for name in names }

# ---- 3. 两两配对 t 检验，得到 p-value 矩阵 ----
n = len(names)
p_mat = np.zeros((n, n))
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        if i == j:
            p_mat[i, j] = np.nan
        else:
            _, p = ttest_rel(errors[ni], errors[nj])
            p_mat[i, j] = p

# ---- 4. 绘制热力图 ----
alpha = 0.05
mask = p_mat > alpha    # 不显著（p>0.05）的地方遮罩

plt.figure(figsize=(6,5))
ax = sns.heatmap(
    p_mat,
    mask=mask,
    cmap="RdYlGn_r",
    vmin=0, vmax=alpha,
    annot=True, fmt=".3f",
    xticklabels=names,
    yticklabels=names,
    cbar_kws={"label": "p-value"}
)

# 在不显著的位置打叉
for i, j in itertools.product(range(n), range(n)):
    if mask[i,j]:
        ax.text(j+0.5, i+0.5, '×', ha='center', va='center', color='black', fontsize=14)

plt.title("Pairwise Significance (paired t-test p-values)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# 计算两模型的绝对误差
err_x    = np.abs(np.array(nbeatsx)     - np.array(actual))
err_xlst = np.abs(np.array(nbeatsx_lstm)- np.array(actual))

# 构造 DataFrame 便于 seaborn
import pandas as pd
df = pd.DataFrame({
    'NBeatsx': err_x,
    'NBeatsx-LSTM': err_xlst
})
df_melt = df.melt(var_name='Model', value_name='AbsError')

plt.figure(figsize=(6,4))
sns.boxplot(x='Model', y='AbsError', data=df_melt)
plt.title('Absolute Error Distribution')
plt.ylabel('Absolute Error')
plt.tight_layout()
plt.show()

diff = err_xlst - err_x  # 正值：NBeatsx-LSTM 较差；负值：NBeatsx 较差

plt.figure(figsize=(8,3))
plt.axhline(0, color='gray', linestyle='--')
plt.plot(dates, diff, marker='o', label='Err(NBeatsx-LSTM) - Err(NBeatsx)')
plt.xticks(rotation=45)
plt.ylabel('Error Difference')
plt.title('Time Series of Error Difference')
plt.legend()
plt.tight_layout()
plt.show()


mean_err = (err_x + err_xlst) / 2
diff_err = err_xlst - err_x

plt.figure(figsize=(5,5))
plt.scatter(mean_err, diff_err, alpha=0.7)
plt.axhline(np.mean(diff_err), color='red', label='Mean Diff')
plt.axhline(np.mean(diff_err) + 1.96*np.std(diff_err),
            color='gray', linestyle='--', label='+1.96 SD')
plt.axhline(np.mean(diff_err) - 1.96*np.std(diff_err),
            color='gray', linestyle='--', label='-1.96 SD')
plt.xlabel('Mean Absolute Error')
plt.ylabel('Difference (LSTMx - NBeatsx)')
plt.title('Bland–Altman Plot')
plt.legend()
plt.tight_layout()
plt.show()


