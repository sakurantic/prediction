# ============================================================
# 代码块 1: 每小时历史均值 → 预测表 df_prediction
# 前提: 上方 cell 已构建好 df_train
#   - 列含 pm_ave (三站 PM2.5 均值)
#   - 索引为 DatetimeIndex
# 本块产出: df_prediction，只有两列 → timestamp / predicted_pm_ave
# ============================================================

# 按小时 (0~23) 分组，对训练集 pm_ave 求均值，得到 24 个基准预测值
# df_train.index 是 DatetimeIndex，.hour 直接取出每个时间点的小时
hourly_baseline = df_train.groupby(df_train.index.hour)['pm_ave'].mean()

print('=== 24 小时基线预测值 (μg/m³) ===')
print(hourly_baseline.round(2).to_string())

# 读取测试集，构造时间戳 (na_values='NA' 把字符串 NA 转成 NaN)
df_test = pd.read_csv('data/ShanghaiPM_Test.csv', na_values='NA')
df_test['timestamp'] = pd.to_datetime(df_test[['year', 'month', 'day', 'hour']])

# 预测表: 时间戳 + 预测值
# 预测值 = 该测试时间点的小时 → 查 hourly_baseline 表得到的历史均值
df_prediction = pd.DataFrame({
    'timestamp': df_test['timestamp'],
    'predicted_pm_ave': df_test['timestamp'].dt.hour.map(hourly_baseline)
})

print(f'\n预测表行数: {len(df_prediction)}')
df_prediction.head(10)


# ============================================================
# 代码块 2: df_prediction 与测试集真实值逐条比对，算均方误差 MSE
# ============================================================

# 从测试集算出每个时间点的真实 pm_ave (三站均值，至少一个有效)
pm_cols = ['PM_Jingan', 'PM_US Post', 'PM_Xuhui']
df_test['pm_ave'] = df_test[pm_cols].mean(axis=1)

# 把真实值并进 df_prediction (按行对齐，两者时间戳顺序一致)
df_prediction['actual_pm_ave'] = df_test['pm_ave'].values

# 逐条比对: 残差 = 真实值 - 预测值 (每个测试点单独算，不先做小时平均)
df_prediction['error'] = df_prediction['actual_pm_ave'] - df_prediction['predicted_pm_ave']

# 只保留有真实值的点参与 loss (三站全缺失的测试点无法评估)
mask = df_prediction['actual_pm_ave'].notna()
err = df_prediction.loc[mask, 'error']

# 均方误差 MSE = 残差平方的均值
mse = (err ** 2).mean()

print(f'有效比对样本数: {mask.sum()} / {len(df_prediction)}')
print(f'MSE = {mse:.2f}')
df_prediction.head(10)
