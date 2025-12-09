# 5种模型对比系统使用指南

## 📦 文件列表

```
compare/
├── SimpleLSTM.py           # 标准LSTM模型
├── RandomForestModel.py    # 随机森林模型
├── AblationPewLSTM.py      # 消融PewLSTM（带开关）
├── modifiedPSTM.py         # 完整PewLSTM（复制）
├── overall.py              # 整合测试系统
├── visualize.py            # 可视化工具
└── usage.md               # 本文档
```

## 🚀 快速开始

### 1. 运行Mini实验 (P1-P10, 1h, departure, 5种模型)

```bash
cd compare
python overall.py --mini --version v1 --epochs 500
```

**输出**:
- 终端显示5种模型的训练进度
- 生成 `results_v1.csv` 包含所有结果
- 生成 `checkpoints/` 文件夹保存断点

### 2. 可视化结果

```bash
# Mini版本可视化 (推荐)
python visualize.py --csv results_v1.csv --mini --output mini_comparison.png

# Accuracy柱状图
python visualize.py --csv results_v1.csv --metric Accuracy --output acc.png

# RMSE柱状图
python visualize.py --csv results_v1.csv --metric RMSE --output rmse.png

# 摘要表格
python visualize.py --csv results_v1.csv --summary --output summary.png

# 热图
python visualize.py --csv results_v1.csv --heatmap --output heatmap.png
```

## 📊 5种模型说明

| 模型 | 说明 | 特点 |
|------|------|------|
| **PewLSTM** | 完整版 | period历史 + weather门控 |
| **SimpleLSTM** | 标准LSTM | 仅停车数据 |
| **RandomForest** | 随机森林 | sklearn实现 |
| **PewLSTM w/o Periodic** | 消融模型 | 禁用h_d,h_w,h_m |
| **PewLSTM w/o Weather** | 消融模型 | 禁用e_t |

## 🎯 命令行参数

### overall.py

```bash
python overall.py [OPTIONS]
```

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--mini` | Mini版本 | - | `--mini` |
| `--full` | 完整版本 | - | `--full` |
| `--version` | 版本标签 | `v1` | `--version test` |
| `--epochs` | 训练轮数 | `500` | `--epochs 100` |
| `--parks` | 停车场 | `all` | `--parks "0,1,2"` |
| `--hours` | 预测时长 | `1` | `--hours 2` |
| `--task` | 任务类型 | `departure` | `--task arrival` |

### visualize.py

```bash
python visualize.py [OPTIONS]
```

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--csv` | CSV文件 | `results_v1.csv` | `--csv results_test.csv` |
| `--metric` | 指标 | `Accuracy` | `--metric RMSE` |
| `--hours` | 时长 | `1h` | `--hours 2h` |
| `--parks` | 停车场 | `all` | `--parks "P1,P2,P3"` |
| `--mini` | Mini可视化 | - | `--mini` |
| `--multi-hours` | 多时长对比 | - | `--multi-hours` |
| `--summary` | 摘要表格 | - | `--summary` |
| `--heatmap` | 热图 | - | `--heatmap` |
| `--output` | 输出文件 | `comparison.png` | `--output result.png` |

## 📈 使用场景

### 场景1: 快速测试 (只测P1)

```bash
# 100 epochs快速验证
python overall.py --mini --version quick --epochs 100 --parks "0"

# 可视化
python visualize.py --csv results_quick.csv --parks "P1" --output quick.png
```

### 场景2: 标准Mini实验 (所有停车场)

```bash
# 1. 运行实验
python overall.py --mini --version v1 --epochs 500

# 2. Mini可视化
python visualize.py --csv results_v1.csv --mini --output mini_v1.png

# 3. 查看摘要
python visualize.py --csv results_v1.csv --summary
```

### 场景3: 2h/3h预测

```bash
# 2h预测
python overall.py --mini --version 2h --epochs 500 --hours 2
python visualize.py --csv results_2h.csv --hours 2h --mini --output 2h.png

# 3h预测
python overall.py --mini --version 3h --epochs 500 --hours 3
python visualize.py --csv results_3h.csv --hours 3h --mini --output 3h.png

# 多时长对比
python visualize.py --csv results_v1.csv --multi-hours --output multi.png
```

### 场景4: 完整实验 (1h/2h/3h × departure/arrival)

```bash
python overall.py --full --version full_v1 --epochs 500
# 这会生成 results_full_v1_complete.csv
```

## 📁 输出文件

### results_v1.csv 格式

```csv
Park,Model,Hours,Task,Accuracy,RMSE
P1,PewLSTM,1h,departure,85.30,2.15
P1,SimpleLSTM,1h,departure,82.50,2.48
P1,RandomForest,1h,departure,80.10,2.95
P1,PewLSTM_w/o_Periodic,1h,departure,83.40,2.35
P1,PewLSTM_w/o_Weather,1h,departure,84.20,2.25
...
```

### Checkpoints

```
checkpoints/
├── PewLSTM_P1_v1_epoch50.pth
├── PewLSTM_P1_v1_epoch100.pth
├── SimpleLSTM_P1_v1_epoch50.pth
└── ...
```

## ⚙️ 关键特性

### 1. 数据划分 (75/25时间序列)

```python
# 非随机划分
train_x = x[:75%]  # 早期数据
test_x = x[75%:]   # 晚期数据
```

### 2. 训练进度条

```
Training PewLSTM: 100%|████████| 500/500 [02:15<00:00, Loss: 0.00234]
```

### 3. 断点保存

- 每50 epochs自动保存
- 训练中断可恢复
- 使用相同version自动Resume

### 4. Accuracy计算

```python
accuracy = (1 - 平均相对误差) × 100%
```

### 5. RMSE计算

```python
rmse = sqrt(平均平方误差)  # 反归一化后
```

## 🔍 预期结果

根据论文和模型架构，预期性能排序：

| 排名 | 模型 | 预期Accuracy |
|-----|------|-------------|
| 🥇 1 | PewLSTM | ~85.3% |
| 🥈 2 | PewLSTM w/o Weather | ~84% |
| 🥉 3 | PewLSTM w/o Periodic | ~83% |
| 4 | SimpleLSTM | ~82% |
| 5 | RandomForest | ~80% |

## ⚠️ 注意事项

1. **运行目录**: 必须在 `compare/` 文件夹内运行
2. **数据路径**: 自动从上级目录加载数据
3. **预训练模型**: P1的1h departure可用预训练模型
4. **训练时间**: 500 epochs × 10停车场 × 5模型 ≈ 1-2小时
5. **内存占用**: Random Forest可能占用较多内存

## 🐛 故障排除

### 问题1: 找不到main.py

```bash
# 确保在compare文件夹内
cd /Users/0ximio/Desktop/PewLSTM_Agy/compare
python overall.py --mini
```

### 问题2: No module named 'tqdm'

```bash
pip install tqdm
```

### 问题3: 找不到预训练模型

```python
# overall.py会自动处理，使用use_pretrained_pewlstm=False训练新模型
```

## 📝 Python API使用

### 自定义实验

```python
from overall import run_mini_experiments

# 运行特定配置
df = run_mini_experiments(
    park_indices=[0, 1, 2],  # P1,P2,P3
    predict_hours=1,
    task='departure',
    version='custom_v1',
    epochs=300
)

print(df)
```

### 自定义可视化

```python
from visualize import load_results, plot_comparison

df = load_results('results_v1.csv')

plot_comparison(
    df,
    metric='Accuracy',
    predict_hours='1h',
    parks=['P1', 'P5', 'P10'],
    save_path='custom.png'
)
```

## 🎓 模型技术细节

### SimpleLSTM
- 标准LSTM门控
- 无周期特征
- 无天气门控

### RandomForest
- sklearn.ensemble.RandomForestRegressor
- n_estimators=100
- max_depth=20
- 数据展平为2D输入

### AblationPewLSTM
- 可选参数: `use_periodic`, `use_weather`
- 动态禁用特定特征
- 用于消融实验
