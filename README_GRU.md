# GRU和PewGRU模型使用指南

## 📁 新增文件

1. **GRU.py** - 标准GRU模型实现
2. **PewGRU.py** - 改进的GRU模型(周期+天气门控)
3. **overall.py** - 整合测试系统
4. **visualize.py** - 可视化工具

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install tqdm  # 进度条库
```

其他依赖已在 `requirements.txt` 中。

### 2. 运行Mini版本测试 (P1-P10, 1h, departure)

```bash
# 基础运行 (500 epochs)
python overall.py --mini --version v1 --epochs 500

# 快速测试 (100 epochs)
python overall.py --mini --version test --epochs 100

# 指定特定停车场
python overall.py --mini --version v1 --epochs 500 --parks "0,1,2"  # 只测试P1,P2,P3
```

**输出**:
- 终端显示训练进度条
- 生成 `results_v1.csv` 包含所有结果
- 生成 `checkpoints/` 文件夹保存模型断点

### 3. 运行完整版本 (所有组合)

```bash
# 完整实验: 1h/2h/3h × departure/arrival × P1-P10
python overall.py --full --version full_v1 --epochs 500

# 只测试特定预测时长
python overall.py --mini --version v2 --epochs 500 --hours 2  # 2h预测

# 测试arrival任务
python overall.py --mini --version v3 --epochs 500 --task arrival
```

### 4. 可视化结果

```bash
# 基础柱状图 (Accuracy)
python visualize.py --csv results_v1.csv --metric Accuracy --hours 1h

# RMSE对比
python visualize.py --csv results_v1.csv --metric RMSE --hours 1h

# 多指标多时长对比
python visualize.py --csv results_v1.csv --multi --output multi_comparison.png

# 显示摘要表格
python visualize.py --csv results_v1.csv --summary

# 生成热图
python visualize.py --csv results_v1.csv --heatmap --metric Accuracy

# 指定特定停车场
python visualize.py --csv results_v1.csv --parks "P1,P2,P3" --output p1_p3.png
```

## 📊 结果文件格式

**results_v1.csv** 格式:
```
Park,Model,Hours,Task,Accuracy,RMSE
P1,PewLSTM,1h,departure,85.30,2.15
P1,GRU,1h,departure,82.50,2.48
P1,PewGRU,1h,departure,84.20,2.25
...
```

## 🔄 断点恢复

```bash
# 训练会自动每50 epochs保存一次断点到 checkpoints/ 文件夹
# 如果训练中断，重新运行相同命令即可恢复（注意使用相同的version）
python overall.py --mini --version v1 --epochs 500  # 自动从最新checkpoint恢复
```

## 📈 命令行参数说明

### overall.py 参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--mini` | 运行mini版本 | - | `--mini` |
| `--full` | 运行完整版本 | - | `--full` |
| `--version` | 版本标签 | `v1` | `--version test` |
| `--epochs` | 训练轮数 | `500` | `--epochs 100` |
| `--parks` | 停车场索引 | `all` | `--parks "0,1,2"` |
| `--hours` | 预测时长 | `1` | `--hours 2` |
| `--task` | 任务类型 | `departure` | `--task arrival` |

### visualize.py 参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--csv` | CSV文件路径 | `results_v1.csv` | `--csv results_test.csv` |
| `--metric` | 指标 | `Accuracy` | `--metric RMSE` |
| `--hours` | 预测时长 | `1h` | `--hours 2h` |
| `--task` | 任务类型 | `departure` | `--task arrival` |
| `--parks` | 停车场 | `all` | `--parks "P1,P2,P3"` |
| `--fill-missing` | 填充缺失值 | - | `--fill-missing` |
| `--multi` | 多指标对比 | - | `--multi` |
| `--summary` | 显示摘要表格 | - | `--summary` |
| `--heatmap` | 生成热图 | - | `--heatmap` |
| `--output` | 输出文件 | `comparison.png` | `--output result.png` |

## 🎯 典型使用流程

### 场景1: 快速测试新模型

```bash
# 1. 快速训练 (100 epochs, 只测试P1)
python overall.py --mini --version quick --epochs 100 --parks "0"

# 2. 查看结果
python visualize.py --csv results_quick.csv --parks "P1"
```

### 场景2: 完整对比实验

```bash
# 1. 运行mini版本 (P1-P10, 1h, departure)
python overall.py --mini --version v1 --epochs 500

# 2. 生成多种可视化
python visualize.py --csv results_v1.csv --metric Accuracy --output acc_1h.png
python visualize.py --csv results_v1.csv --metric RMSE --output rmse_1h.png
python visualize.py --csv results_v1.csv --summary
python visualize.py --csv results_v1.csv --heatmap
```

### 场景3: 2h和3h预测

```bash
# 1. 运行2h预测
python overall.py --mini --version 2h_test --epochs 500 --hours 2

# 2. 运行3h预测
python overall.py --mini --version 3h_test --epochs 500 --hours 3

# 3. 对比可视化
python visualize.py --csv results_2h_test.csv --hours 2h --output 2h_comparison.png
python visualize.py --csv results_3h_test.csv --hours 3h --output 3h_comparison.png
```

### 场景4: Arrival预测

```bash
# 1. 运行arrival任务
python overall.py --mini --version arrival_v1 --epochs 500 --task arrival

# 2. 可视化
python visualize.py --csv results_arrival_v1.csv --task arrival
```

## 📁 文件结构

运行后生成的文件:

```
PewLSTM_Agy/
├── GRU.py                    # 新增1
├── PewGRU.py                 # 新增2
├── overall.py                # 新增3
├── visualize.py              # 新增4
├── checkpoints/              # 训练断点
│   ├── GRU_P1_v1_epoch50.pth
│   ├── PewGRU_P1_v1_epoch50.pth
│   └── ...
├── results_v1.csv            # 结果CSV
├── comparison.png            # 对比图
├── multi_comparison.png      # 多指标对比图
├── summary_table.png         # 摘要表格
└── heatmap.png               # 热图
```

## 🔍 结果解读

**Accuracy**: 
- 范围: 0-100%
- 计算: `(1 - 平均相对误差) × 100`
- 越高越好
- PewLSTM论文报告: 85.3%

**RMSE**:
- 范围: > 0
- 单位: 车辆数
- 越低越好
- 反映实际预测偏差多少辆车

## ⚠️ 注意事项

1. **训练时间**: 500 epochs × 10停车场 × 3模型 ≈ 30-60分钟
2. **内存占用**: 如果内存不足，减少epochs或分批运行
3. **预训练模型**: P1的1h departure有预训练模型 `model_P1_1h.pth`
4. **数据划分**: 使用75/25时间序列划分（非随机）
5. **缺失值**: 如果某个组合没有数据，可视化会显示0或用`--fill-missing`填充NaN

## 🐛 故障排除

**问题1**: `ModuleNotFoundError: No module named 'tqdm'`
```bash
pip install tqdm
```

**问题2**: `FileNotFoundError: model_P1_1h.pth`
- 确保在项目根目录运行
- 或设置 `use_pretrained_pewlstm=False`

**问题3**: 训练进度条不显示
- 确保终端支持tqdm
- 或使用 `--epochs 10` 快速测试

**问题4**: CUDA out of memory
- 使用CPU训练（默认）
- 或分批运行: `--parks "0,1,2"` 然后 `--parks "3,4,5"` 等

## 📝 代码示例

### 自定义训练脚本

```python
from overall import run_mini_experiments

# 运行特定配置
df = run_mini_experiments(
    park_indices=[0, 1, 2],  # P1, P2, P3
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

# 加载结果
df = load_results('results_v1.csv')

# 自定义绘图
plot_comparison(
    df, 
    metric='Accuracy',
    predict_hours='1h',
    parks=['P1', 'P2', 'P5'],
    save_path='custom_plot.png'
)
```

## 🎓 模型对比说明

| 模型 | 特点 | 参数量 | 训练速度 |
|------|------|--------|---------|
| **PewLSTM** | LSTM + 周期 + 天气 | 最多 | 最慢 |
| **GRU** | 标准GRU | 中等 | 最快 |
| **PewGRU** | GRU + 周期 + 天气 | 中等 | 中等 |

**预期结果**:
- PewLSTM: 最高Accuracy (论文85.3%)
- PewGRU: 接近PewLSTM (预计83-85%)
- GRU: 基线 (预计80-82%)

## 📧 支持

如有问题，请检查:
1. 依赖是否完整安装
2. 数据文件是否存在
3. 命令行参数是否正确

祝实验顺利！🎉
