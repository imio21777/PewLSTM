# GRU和PewGRU模型使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install tqdm
```

### 2. 运行测试

#### Mini版本 (P1-P10, 1h, departure)

```bash
# 基础运行 (500 epochs)
python overall.py --mini --version v1 --epochs 500

# 快速测试 (100 epochs)
python overall.py --mini --version test --epochs 100

# 指定停车场 (只测P1,P2,P3)
python overall.py --mini --version v1 --epochs 500 --parks "0,1,2"
```

#### 完整版本 (所有组合)

```bash
# 1h/2h/3h × departure/arrival × P1-P10
python overall.py --full --version full_v1 --epochs 500
```

#### 自定义配置

```bash
# 2h预测
python overall.py --mini --version v2 --epochs 500 --hours 2

# 3h预测
python overall.py --mini --version v3 --epochs 500 --hours 3

# Arrival任务
python overall.py --mini --version v4 --epochs 500 --task arrival
```

### 3. 可视化结果

```bash
# Accuracy柱状图
python visualize.py --csv results_v1.csv --metric Accuracy --hours 1h

# RMSE柱状图
python visualize.py --csv results_v1.csv --metric RMSE --hours 1h

# 多指标对比 (1h/2h/3h)
python visualize.py --csv results_v1.csv --multi --output multi_comparison.png

# 摘要表格
python visualize.py --csv results_v1.csv --summary

# 热图
python visualize.py --csv results_v1.csv --heatmap --metric Accuracy

# 指定停车场
python visualize.py --csv results_v1.csv --parks "P1,P2,P3" --output p1_p3.png
```

## 命令行参数

### overall.py

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--mini` | 运行mini版本 | - | `--mini` |
| `--full` | 运行完整版本 | - | `--full` |
| `--version` | 版本标签 | `v1` | `--version test` |
| `--epochs` | 训练轮数 | `500` | `--epochs 100` |
| `--parks` | 停车场索引 | `all` | `--parks "0,1,2"` |
| `--hours` | 预测时长 (1/2/3) | `1` | `--hours 2` |
| `--task` | 任务类型 | `departure` | `--task arrival` |

### visualize.py

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--csv` | CSV文件路径 | `results_v1.csv` | `--csv results_test.csv` |
| `--metric` | 指标 | `Accuracy` | `--metric RMSE` |
| `--hours` | 预测时长 | `1h` | `--hours 2h` |
| `--task` | 任务类型 | `departure` | `--task arrival` |
| `--parks` | 停车场 | `all` | `--parks "P1,P2,P3"` |
| `--multi` | 多指标对比 | - | `--multi` |
| `--summary` | 摘要表格 | - | `--summary` |
| `--heatmap` | 热图 | - | `--heatmap` |
| `--output` | 输出文件 | `comparison.png` | `--output result.png` |

## 典型工作流

### 场景1: 快速验证

```bash
# 1. 快速训练 (100 epochs, 只测P1)
python overall.py --mini --version quick --epochs 100 --parks "0"

# 2. 查看结果
python visualize.py --csv results_quick.csv --parks "P1"
```

### 场景2: 标准实验

```bash
# 1. 运行mini版本
python overall.py --mini --version v1 --epochs 500

# 2. 生成可视化
python visualize.py --csv results_v1.csv --metric Accuracy --output acc.png
python visualize.py --csv results_v1.csv --metric RMSE --output rmse.png
python visualize.py --csv results_v1.csv --summary
```

### 场景3: 多时长预测对比

```bash
# 1. 分别运行1h/2h/3h
python overall.py --mini --version 1h --epochs 500 --hours 1
python overall.py --mini --version 2h --epochs 500 --hours 2
python overall.py --mini --version 3h --epochs 500 --hours 3

# 2. 对比可视化
python visualize.py --csv results_1h.csv --hours 1h --output 1h.png
python visualize.py --csv results_2h.csv --hours 2h --output 2h.png
python visualize.py --csv results_3h.csv --hours 3h --output 3h.png
```

## 结果文件

### results_v1.csv 格式

```csv
Park,Model,Hours,Task,Accuracy,RMSE
P1,PewLSTM,1h,departure,85.30,2.15
P1,GRU,1h,departure,82.50,2.48
P1,PewGRU,1h,departure,84.20,2.25
...
```

### 生成的文件

```
PewLSTM_Agy/
├── checkpoints/              # 训练断点
│   ├── GRU_P1_v1_epoch50.pth
│   └── PewGRU_P1_v1_final.pth
├── results_v1.csv            # 结果CSV
├── comparison.png            # 对比柱状图
├── multi_comparison.png      # 多指标对比
├── summary_table.png         # 摘要表格
└── heatmap.png               # 热图
```

## 断点恢复

训练会自动每50 epochs保存断点到 `checkpoints/` 文件夹。

如果训练中断，重新运行相同命令即可自动恢复：

```bash
# 训练到200 epochs时中断
python overall.py --mini --version v1 --epochs 500
# Ctrl+C 中断

# 重新运行，自动从最新checkpoint恢复
python overall.py --mini --version v1 --epochs 500
```

## 注意事项

1. **训练时间**: 500 epochs × 10停车场 × 3模型 ≈ 30-60分钟
2. **数据划分**: 75%训练 / 25%测试 (时间序列划分)
3. **预训练模型**: P1的1h departure有预训练模型 `model_P1_1h.pth`
4. **评估指标**:
   - **Accuracy**: (1 - 平均相对误差) × 100%
   - **RMSE**: 反归一化后的均方根误差 (单位: 辆)

## 故障排除

### PyTorch导入错误

```bash
pip uninstall torch
pip install torch
```

### tqdm未安装

```bash
pip install tqdm
```

### 找不到数据文件

确保在项目根目录运行：
```bash
cd /Users/0ximio/Desktop/PewLSTM_Agy
python overall.py --mini --version v1
```
