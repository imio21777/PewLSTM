# run_modified_pewlstm.py 使用指南

## 前置依赖
- 已安装 `torch`（若当前环境导入失败，请重新安装或换新的 PyTorch 版本）。
- 已安装 `pandas` 与 `scikit-learn`。
- 若希望看到训练进度条，请额外安装 `tqdm`。

## 运行前准备
请在项目根目录的 `PewLSTM` 文件夹内执行命令：
```bash
cd PewLSTM
python run_modified_pewlstm.py --help
```

## 命令格式
```bash
python run_modified_pewlstm.py [参数]
```
- `--lots [LOTS ...]`：指定要训练的停车场编号，如 `--lots 1 3 5`。默认会遍历所有停车场（P1–P10）。
- `--train-ratio TRAIN_RATIO`：训练集占总天序列的比例，默认 `0.8`。
- `--epochs EPOCHS`：训练轮数，默认 `200`。
- `--lr LR`：Adam 学习率，默认 `1e-3`。
- `--weight-decay WEIGHT_DECAY`：Adam 权重衰减系数，默认 `0.0`。
- `--seed SEED`：随机数种子，默认 `42`。

## 示例
```bash
python run_modified_pewlstm.py --train-ratio 0.8 --epochs 200 --lots 1 3
```
含义：
- 将停车场 P1 和 P3 的数据按天切分并按 8:2 划分训练/测试集；
- 训练 200 个 epoch；
- 训练完成后会输出 RMSE、MAE、MAPE 以及最终的训练/测试损失情况。

若想快速验证流程，可先减小轮数，例如：
```bash
python run_modified_pewlstm.py --epochs 5 --lots 1
```
