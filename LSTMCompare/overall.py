"""
Overall Testing System for 5 Model Comparison
支持 5 种模型对比:
1. PewLSTM (完整版)
2. Simple LSTM (仅停车数据)
3. Random Forest (随机森林)
4. PewLSTM w/o Periodic (无周期)
5. PewLSTM w/o Weather (无天气)
"""

import sys
import os

# 在导入main之前，先切换到父目录
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import argparse
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import models (from compare folder)
from SimpleLSTM import SimpleLSTM
from RandomForestModel import RandomForestModel
from AblationPewLSTM import AblationPewLSTM
from modifiedPSTM import pew_LSTM

# Import data loading functions (from parent folder)
from main import pGetAllData, p2GetAllData

# Configuration
HIDDEN_DIM = 1
SEQ_SIZE = 24


def prepare_data(x, y, predict_hours=1):
    """
    准备数据 (方案B: 直接修改label)
    Args:
        x: [hour_size, 5]
        y: [hour_size]
        predict_hours: 预测时长 (1/2/3)
    Returns:
        x: [days, 24, 5]
        y: [days * 24]
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    # 重塑为天级批次
    x = x[:((x.size(0) // 24) * 24)].reshape((x.size(0) // 24, 24, 5))
    y = y[:((y.size(0) // 24) * 24)]
    
    # 对于2h/3h预测,移位label
    if predict_hours > 1:
        shift = predict_hours - 1
        y = y[shift:]
        x = x[:len(y) // 24]
        y = y[:len(x) * 24]
    
    return x, y


def split_train_test(x, y, train_ratio=0.75):
    """
    时间序列划分 (75/25)
    """
    l = int(train_ratio * len(x))
    train_x = x[:l]
    train_y = y[:l*24]
    test_x = x[l:]
    test_y = y[l*24:]
    
    return train_x, train_y, test_x, test_y


# ========== 模型包装类 ==========

class SimpleLSTMModel(nn.Module):
    """Simple LSTM包装"""
    def __init__(self):
        super(SimpleLSTMModel, self).__init__()
        self.lstm1 = SimpleLSTM(1, HIDDEN_DIM)
        self.lstm2 = SimpleLSTM(HIDDEN_DIM, HIDDEN_DIM)
        self.fc = nn.Linear(HIDDEN_DIM, 1)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, input):
        x_input = input[:, :, -1].unsqueeze(2)  # 只用停车数
        
        h1, c1 = self.lstm1(x_input)
        h2, c2 = self.lstm2(h1)
        out = h2.contiguous().view(-1, HIDDEN_DIM)
        out = self.fc(out).view(-1)
        return out


class PewLSTMModel(nn.Module):
    """PewLSTM包装"""
    def __init__(self):
        super(PewLSTMModel, self).__init__()
        self.lstm1 = pew_LSTM(1, HIDDEN_DIM, 4)
        self.lstm2 = pew_LSTM(HIDDEN_DIM, HIDDEN_DIM, 4)
        self.fc = nn.Linear(HIDDEN_DIM, 1)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, input):
        x_weather = input[:, :, :-1]
        x_input = input[:, :, -1].unsqueeze(2)
        
        h1, c1 = self.lstm1(x_input, x_weather)
        h2, c2 = self.lstm2(h1, x_weather)
        out = h2.contiguous().view(-1, HIDDEN_DIM)
        out = self.fc(out).view(-1)
        return out


class AblationPewLSTMModel(nn.Module):
    """消融PewLSTM包装"""
    def __init__(self, use_periodic=True, use_weather=True):
        super(AblationPewLSTMModel, self).__init__()
        self.lstm1 = AblationPewLSTM(1, HIDDEN_DIM, 4, use_periodic, use_weather)
        self.lstm2 = AblationPewLSTM(HIDDEN_DIM, HIDDEN_DIM, 4, use_periodic, use_weather)
        self.fc = nn.Linear(HIDDEN_DIM, 1)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, input):
        x_weather = input[:, :, :-1]
        x_input = input[:, :, -1].unsqueeze(2)
        
        h1, c1 = self.lstm1(x_input, x_weather)
        h2, c2 = self.lstm2(h1, x_weather)
        out = h2.contiguous().view(-1, HIDDEN_DIM)
        out = self.fc(out).view(-1)
        return out


# ========== 训练函数 ==========

def train_model(model, train_x, train_y, epochs=500, lr=1e-2,
                model_name='model', version='v1', checkpoint_dir='./checkpoints',
                save_interval=50):
    """
    训练模型(带进度条和断点保存)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    start_epoch = 0
    # Try to load latest checkpoint
    checkpoints = glob.glob(f'{checkpoint_dir}/{model_name}_{version}_epoch*.pth')
    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_epoch')[1].split('.pth')[0]))
        print(f"  -> Found checkpoint: {latest_ckpt}")
        try:
            checkpoint = torch.load(latest_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"  -> Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"  -> Failed to load checkpoint: {e}")

    progress_bar = tqdm(range(start_epoch, epochs), desc=f'Training {model_name}')
    
    for epoch in progress_bar:
        model.train()
        out = model(train_x)
        loss = loss_fn(out, train_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'Loss': f'{loss.item():.5f}'})
        
        # 保存断点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'{checkpoint_dir}/{model_name}_{version}_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)
    
    # 保存最终模型
    final_path = f'{checkpoint_dir}/{model_name}_{version}_final.pth'
    torch.save(model.state_dict(), final_path)
    
    return model


def train_random_forest(model, train_x, train_y):
    """训练随机森林"""
    print("  Training Random Forest...")
    model.train(train_x.numpy(), train_y.numpy())
    print("  ✓ Training complete")
    return model


# ========== 评估函数 ==========

def evaluate_model(model, test_x, test_y, scaler, is_rf=False):
    """
    评估模型 (Accuracy + RMSE)
    """
    if is_rf:
        # Random Forest
        pred_test = model.predict(test_x.numpy()).reshape(-1, 1)
        test_y_numpy = test_y.reshape(-1, 1).numpy()
    else:
        # LSTM模型
        model.eval()
        with torch.no_grad():
            pred_test = model(test_x).cpu().detach().numpy().reshape(-1, 1)
        test_y_numpy = test_y.reshape(-1, 1).cpu().numpy()
    
    # 反归一化 (RMSE)
    atest_y = scaler.inverse_transform(test_y_numpy)
    apred_test = scaler.inverse_transform(pred_test)
    
    # Accuracy (归一化值)
    p = 0.0
    k = 0
    for i in range(len(test_y_numpy) - 1):
        k += 1
        if test_y_numpy[i] != 0:
            p += abs(test_y_numpy[i] - pred_test[i+1]) / test_y_numpy[i]
        else:
            p += abs(test_y_numpy[i] - pred_test[i+1])
    
    accuracy = (1 - p/k) * 100 if k > 0 else 0
    
    # RMSE (反归一化值)
    r = 0
    for i in range(len(test_y_numpy) - 1):
        r += (atest_y[i] - apred_test[i+1])**2
    
    rmse = sqrt(r/k) if k > 0 else 0
    
    return accuracy[0], rmse


# ========== Mini实验 ==========

def run_mini_experiments(park_indices=range(10), predict_hours=1,
                        task='departure', version='v1', epochs=500,
                        use_pretrained_pewlstm=True):
    """
    Mini实验: P1-P10, 1h, departure, 5种模型
    """
    results = []
    
    for park_idx in park_indices:
        print(f"\n{'='*60}")
        print(f"  Testing Park P{park_idx+1} | {predict_hours}h | {task}")
        print(f"{'='*60}")
        
        # 加载数据
        if task == 'departure':
            x, y, s = pGetAllData(park_idx)
        else:
            x, y, s = p2GetAllData(park_idx)
        
        x = x.astype('float32')
        y = y.astype('float32')
        
        # 准备数据
        x, y = prepare_data(x, y, predict_hours)
        train_x, train_y, test_x, test_y = split_train_test(x, y)
        
        print(f"  Data: {len(x)} days ({len(train_x)} train, {len(test_x)} test)")
        
        # ===== 1. PewLSTM =====
        if use_pretrained_pewlstm and predict_hours == 1 and task == 'departure':
            print(f"\n  [1/5] PewLSTM (using pretrained model)")
            pewlstm_model = PewLSTMModel()
            pewlstm_model.load_state_dict(torch.load('model_P1_1h.pth'))
        else:
            print(f"\n  [1/5] PewLSTM (training)")
            pewlstm_model = PewLSTMModel()
            pewlstm_model = train_model(pewlstm_model, train_x, train_y, epochs=epochs,
                                       model_name=f'PewLSTM_P{park_idx+1}', version=version)
        
        acc, rmse = evaluate_model(pewlstm_model, test_x, test_y, s)
        results.append({
            'Park': f'P{park_idx+1}',
            'Model': 'PewLSTM',
            'Hours': f'{predict_hours}h',
            'Task': task,
            'Accuracy': round(acc, 2),
            'RMSE': round(rmse, 2)
        })
        print(f"  ✓ PewLSTM - Accuracy: {acc:.2f}%, RMSE: {rmse:.2f}")
        
        # ===== 2. Simple LSTM =====
        print(f"\n  [2/5] Simple LSTM (training)")
        simple_lstm = SimpleLSTMModel()
        simple_lstm = train_model(simple_lstm, train_x, train_y, epochs=epochs,
                                  model_name=f'SimpleLSTM_P{park_idx+1}', version=version)
        
        acc, rmse = evaluate_model(simple_lstm, test_x, test_y, s)
        results.append({
            'Park': f'P{park_idx+1}',
            'Model': 'SimpleLSTM',
            'Hours': f'{predict_hours}h',
            'Task': task,
            'Accuracy': round(acc, 2),
            'RMSE': round(rmse, 2)
        })
        print(f"  ✓ SimpleLSTM - Accuracy: {acc:.2f}%, RMSE: {rmse:.2f}")
        
        # ===== 3. Random Forest =====
        print(f"\n  [3/5] Random Forest")
        rf_model = RandomForestModel(n_estimators=100, max_depth=20)
        rf_model = train_random_forest(rf_model, train_x, train_y)
        
        acc, rmse = evaluate_model(rf_model, test_x, test_y, s, is_rf=True)
        results.append({
            'Park': f'P{park_idx+1}',
            'Model': 'RandomForest',
            'Hours': f'{predict_hours}h',
            'Task': task,
            'Accuracy': round(acc, 2),
            'RMSE': round(rmse, 2)
        })
        print(f"  ✓ RandomForest - Accuracy: {acc:.2f}%, RMSE: {rmse:.2f}")
        
        # ===== 4. PewLSTM w/o Periodic =====
        print(f"\n  [4/5] PewLSTM w/o Periodic (training)")
        pewlstm_no_periodic = AblationPewLSTMModel(use_periodic=False, use_weather=True)
        pewlstm_no_periodic = train_model(pewlstm_no_periodic, train_x, train_y, epochs=epochs,
                                         model_name=f'PewLSTM_NoPeriodic_P{park_idx+1}', version=version)
        
        acc, rmse = evaluate_model(pewlstm_no_periodic, test_x, test_y, s)
        results.append({
            'Park': f'P{park_idx+1}',
            'Model': 'PewLSTM_w/o_Periodic',
            'Hours': f'{predict_hours}h',
            'Task': task,
            'Accuracy': round(acc, 2),
            'RMSE': round(rmse, 2)
        })
        print(f"  ✓ PewLSTM w/o Periodic - Accuracy: {acc:.2f}%, RMSE: {rmse:.2f}")
        
        # ===== 5. PewLSTM w/o Weather =====
        print(f"\n  [5/5] PewLSTM w/o Weather (training)")
        pewlstm_no_weather = AblationPewLSTMModel(use_periodic=True, use_weather=False)
        pewlstm_no_weather = train_model(pewlstm_no_weather, train_x, train_y, epochs=epochs,
                                        model_name=f'PewLSTM_NoWeather_P{park_idx+1}', version=version)
        
        acc, rmse = evaluate_model(pewlstm_no_weather, test_x, test_y, s)
        results.append({
            'Park': f'P{park_idx+1}',
            'Model': 'PewLSTM_w/o_Weather',
            'Hours': f'{predict_hours}h',
            'Task': task,
            'Accuracy': round(acc, 2),
            'RMSE': round(rmse, 2)
        })
        print(f"  ✓ PewLSTM w/o Weather - Accuracy: {acc:.2f}%, RMSE: {rmse:.2f}")
    
    # 保存结果
    df = pd.DataFrame(results)
    result_path = f'results_{version}.csv'
    df.to_csv(result_path, index=False)
    print(f"\n{'='*60}")
    print(f"  Results saved to: {result_path}")
    print(f"{'='*60}\n")
    
    return df


# ========== 完整实验 ==========

def run_full_experiments(park_indices=range(10),
                        predict_hours_list=[1, 2, 3],
                        tasks=['departure', 'arrival'],
                        version='full_v1', epochs=500):
    """完整实验"""
    all_results = []
    
    for hours in predict_hours_list:
        for task in tasks:
            print(f"\n{'#'*60}")
            print(f"#  Prediction: {hours}h | Task: {task}")
            print(f"{'#'*60}")
            
            df = run_mini_experiments(
                park_indices=park_indices,
                predict_hours=hours,
                task=task,
                version=f'{version}_{hours}h_{task}',
                epochs=epochs,
                use_pretrained_pewlstm=False
            )
            all_results.append(df)
    
    # 合并结果
    final_df = pd.concat(all_results, ignore_index=True)
    final_path = f'results_{version}_complete.csv'
    final_df.to_csv(final_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE!")
    print(f"  Final results saved to: {final_path}")
    print(f"{'='*60}\n")
    
    return final_df


# ========== 主程序 ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='5 Model Comparison System')
    parser.add_argument('--mini', action='store_true', help='Run mini version')
    parser.add_argument('--full', action='store_true', help='Run full version')
    parser.add_argument('--version', type=str, default='v1', help='Version tag')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--parks', type=str, default='all', help='Park indices')
    parser.add_argument('--hours', type=int, default=1, help='Prediction hours (1/2/3)')
    parser.add_argument('--task', type=str, default='departure', choices=['departure', 'arrival'])
    parser.add_argument('--train_pew', action='store_true', help='Train PewLSTM instead of using pretrained model in mini mode')
    
    args = parser.parse_args()
    
    # Parse park indices
    if args.parks == 'all':
        park_indices = range(10)
    else:
        park_indices = [int(x) for x in args.parks.split(',')]
    
    if args.mini:
        print("Running MINI experiments (5 models)...")
        run_mini_experiments(
            park_indices=park_indices,
            predict_hours=args.hours,
            task=args.task,
            version=args.version,
            epochs=args.epochs,
            use_pretrained_pewlstm=not args.train_pew
        )
    elif args.full:
        print("Running FULL experiments...")
        run_full_experiments(
            park_indices=park_indices,
            version=args.version,
            epochs=args.epochs
        )
    else:
        print("Please specify --mini or --full")
        print("Example: python overall.py --mini --version v1 --epochs 500")
