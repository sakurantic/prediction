# ============================================================
# 基线预测模型 —— 每小时历史均值周期预测
# 思路：历史上所有 0 点数据的平均 pm2.5 值 → 就是 0 点的预测值
#       历史上所有 1 点数据的平均 pm2.5 值 → 就是 1 点的预测值
#       ...以此类推，得到一个 24 步的周期信号
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 步骤 1: 用训练集构建 pm_ave ---
# 重新读取训练数据（保持独立，不依赖前面 cell 的 df_raw）
df_train = pd.read_csv('data/ShanghaiPM_Training.csv', na_values='NA')

# 拼接时间戳列
df_train['timestamp'] = pd.to_datetime(
    df_train[['year', 'month', 'day', 'hour']]
)

# 三个 PM 监测站的列名
pm_cols = ['PM_Jingan', 'PM_US Post', 'PM_Xuhui']

# 计算 pm_ave: 三站均值（axis=1 逐行求平均，skipna=True 自动跳过 NaN）
# 注意：如果某行的三站值全为 NaN，则 pm_ave 也为 NaN
df_train['pm_ave'] = df_train[pm_cols].mean(axis=1)

# 删除三站均缺失的行（pm_ave 为 NaN 的行）
df_train = df_train.dropna(subset=['pm_ave'])

# --- 步骤 2: 按小时分组，计算历史均值 ---
df_train['hour'] = df_train['timestamp'].dt.hour  # 提取小时 0~23

# groupby('hour') → 把数据按 0~23 分成 24 组
# ['pm_ave'].mean() → 每组求均值 → 得到 24 个预测基准值
hourly_baseline = df_train.groupby('hour')['pm_ave'].mean()

print('=== 24 小时基线预测值 (μg/m³) ===')
print(hourly_baseline.round(2).to_string())

# --- 步骤 3: 在测试集上生成预测 df_prediction ---
# 读取测试集，构造时间戳
df_test = pd.read_csv('data/ShanghaiPM_Test.csv')
df_test['timestamp'] = pd.to_datetime(df_test[['year', 'month', 'day', 'hour']])

# 为每个测试时间点，根据其小时查表得到历史均值作为预测值
df_prediction = pd.DataFrame({
    'timestamp': df_test['timestamp'],
    'predicted_pm_ave': df_test['timestamp'].dt.hour.map(hourly_baseline)
})

print(f'\n测试集共 {len(df_prediction)} 个时间点，已全部完成预测')
print(f'预测值标准差: {df_prediction["predicted_pm_ave"].std():.2f}  |  '
      f'均值: {df_prediction["predicted_pm_ave"].mean():.2f}')
df_prediction.head(10)

# --- 步骤 4: 用 matplotlib 绘制预测曲线 ---
# 预期效果: 每 24 小时重复的锯齿波
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    df_prediction['timestamp'],
    df_prediction['predicted_pm_ave'],
    linewidth=0.8,
    color='steelblue',
    label='Baseline (24h hourly mean)'
)

# --- 图表美化 ---
ax.set_xlabel('Timestamp')
ax.set_ylabel('Predicted PM2.5 (μg/m³)')
ax.set_title('Baseline Prediction — 24-Hour Repeating Pattern')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate()          # 自动旋转 x 轴标签，避免重叠
plt.tight_layout()           # 自动调整边距
plt.show()
