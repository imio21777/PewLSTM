# Parking Prediction Method References

本文整理了在 PewLSTM 项目中需要对比的五种停车行为预测方法，对应的论文来源与（若已公开）代码仓库链接如下。

## 1. PewLSTM（含周期与天气感知门控）
- **论文**：Feng Zhang, Ningxuan Feng, Yani Liu, Cheng Yang, Jidong Zhai, Shuhao Zhang, Bingsheng He, Jiazao Lin, Xiaoyong Du. *PewLSTM: Periodic LSTM with Weather-Aware Gating Mechanism for Parking Behavior Prediction*. IJCAI 2020. [PDF](https://www.ijcai.org/proceedings/2020/0610)
- **代码**：官方开源实现（Python + PyTorch）托管于 GitHub。<https://github.com/NingxuanFeng/PewLSTM>
- **说明**：论文中提出的全量模型，融合历史停车周期特征与天气感知门控，是本项目主干实现。

## 2. Simple LSTM（仅基于停车记录）
- **理论基础论文**：Sepp Hochreiter, Jürgen Schmidhuber. *Long Short-Term Memory*. Neural Computation, 1997. [PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)
- **参考实现**：Keras 官方 LSTM 层实现可作为纯停车记录基线的起点。<https://github.com/keras-team/keras/blob/master/keras/layers/rnn/lstm.py>
- **说明**：与 PewLSTM 相比不引入周期和天气特征，仅利用停车时间序列；在 IJCAI 2020 论文中作为对照基线。

## 3. Regression Method（Feng et al., 2019）
- **论文**：Ningxuan Feng, Feng Zhang, Jiazao Lin, Jidong Zhai, Xiaoyong Du. *Statistical Analysis and Prediction of Parking Behavior*. IFIP NPC 2019. [Link](https://link.springer.com/chapter/10.1007/978-3-030-30709-7_8)
- **代码**：截至目前未找到公开仓库；论文描述使用线性回归、岭回归、Lasso、决策树及随机森林等统计学习方法，其中随机森林表现最佳。
- **说明**：论文提出的回归基线仅提供 1 小时预测结果，PewLSTM 论文对其进行了三小时扩展实验。
- **论文**：[Feng et al., 2019] Ningxuan Feng, Feng Zhang, Jiazao Lin, Jidong Zhai, and Xiaoyong Du. Statistical analysis and
prediction of parking behavior. In IFIP International Conference on Network and Parallel Computing, 2019.

## 4. PewLSTM（无周期特征）
- **论文**：与完整的 PewLSTM 相同（IJCAI 2020），该模型是论文中的 ablation 实验“PewLSTM (w/o periodic)”。
- **代码**：可基于官方仓库在配置中屏蔽周期分支实现；仓库中 `modifiedPSTM.py` 已包含用于开关周期门控的实现。<https://github.com/NingxuanFeng/PewLSTM>
- **说明**：通过去掉周期性隐藏状态连接评估周期特征贡献。

## 5. PewLSTM（无天气信息）
- **论文**：同样源自 IJCAI 2020 PewLSTM 论文的 ablation“PewLSTM (w/o weather)”。
- **代码**：在 PewLSTM 官方仓库中禁用天气输入即可复现；本项目更新的 `modifiedPSTM.py` 支持以参数控制。<https://github.com/NingxuanFeng/PewLSTM>
- **说明**：用于衡量天气信息对预测准确度的影响。

> 若后续发现新的官方实现或数据仓库，可在此文件补充链接或备注。

