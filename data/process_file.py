import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # 必须导入以启用 IterativeImputer
from sklearn.impute import IterativeImputer, KNNImputer

# 步骤1：读取数据
df = pd.read_excel('weather.xlsx')

# 步骤2：将 'Day(Local_Date)' 转换为 datetime
# 注意 format='%Y/%m/%d' 要与实际数据格式匹配
df['Day(Local_Date)'] = pd.to_datetime(df['Day(Local_Date)'],
                                       format='%Y/%m/%d',
                                       errors='coerce')

# 如果你的日期有时是 '1999/11/3'、有时是 '1999/11/03'，可尝试不加 format，让 Pandas 自动解析:
# df['Day(Local_Date)'] = pd.to_datetime(df['Day(Local_Date)'], errors='coerce')

# 步骤3：将 '-' 替换为 NaN
df.replace('-', np.nan, inplace=True)

# 步骤4：将数值列转为 float（排除日期列）
# 假设日期列名是 'Day(Local_Date)'
numeric_cols = df.columns.drop('Day(Local_Date)')

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 步骤5：选择数值型列
numeric_cols = df.select_dtypes(include=[np.number]).columns
'''
# 步骤5：用 IterativeImputer 对数值列进行插补
imputer_iter = IterativeImputer(random_state=0, max_iter=40, tol=1e-3)
df_imputed = df.copy()

df_imputed[numeric_cols] = imputer_iter.fit_transform(df_imputed[numeric_cols])

# 如果想保留一定的小数精度，可以 round
df_imputed[numeric_cols] = df_imputed[numeric_cols].round(2)
'''
'''
# 方法1：SimpleImputer
imputer_mean = SimpleImputer(strategy='mean')
df_simple_imputed = df.copy()
df_simple_imputed[numeric_cols] = imputer_mean.fit_transform(df_simple_imputed[numeric_cols]).round(2)
df_simple_imputed[non_numeric_cols] = df[non_numeric_cols]
'''
# 方法2：KNNImputer
imputer_knn = KNNImputer(n_neighbors=1)
df_knn_imputed = df.copy()
df_knn_imputed[numeric_cols] = imputer_knn.fit_transform(df_knn_imputed[numeric_cols])
df_knn_imputed[numeric_cols] = df_knn_imputed[numeric_cols].round(2)




# 步骤7：根据日期生成季节（南半球）
def get_season(date_str):
    # 解析日期字符串为 datetime 对象
    try:
        date = pd.to_datetime(date_str)
        month = date.month
        if month in [12, 1, 2]:
            return 'Summer'   # 夏季
        elif month in [3, 4, 5]:
            return 'Autumn'  # 秋季
        elif month in [6, 7, 8]:
            return 'Winter'  # 冬季
        elif month in [9, 10, 11]:
            return 'Spring'  # 春季
    except:
        return np.nan  # 如果日期解析失败，返回 NaN

df_knn_imputed['Season'] = df_knn_imputed['Day(Local_Date)'].apply(get_season)

# 步骤7：保存结果
df_knn_imputed.to_csv('imputed_data_KNN_1.csv', index=False)
