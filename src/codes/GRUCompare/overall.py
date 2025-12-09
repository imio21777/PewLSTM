"""
Overall Testing System for PewLSTM, GRU, and PewGRU models
Supports: 
- Multiple parking lots (P1-P10)
- Multiple prediction horizons (1h, 2h, 3h)
- Multiple tasks (departure, arrival)
- Training with progress visualization
- Checkpoint saving and resuming
- Result export to CSV
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import argparse
import sys
import os
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 在导入main之前，先切换到父目录
# 在导入main之前，先切换到父目录
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

# Import models
from modifiedPSTM import pew_LSTM
from GRU import standard_GRU
from PewGRU import pew_GRU

# Import data loading functions from main.py
import sys
sys.path.append(os.path.dirname(__file__))
from main import pGetAllData, p2GetAllData, HIDDEN_DIM

# Configuration
HIDDEN_DIM = 1
SEQ_SIZE = 24


def prepare_data(x, y, predict_hours=1):
    """
    Prepare data for training/testing
    Args:
        x: input features [hour_size, 5]
        y: labels [hour_size]
        predict_hours: 1h, 2h, or 3h prediction
    Returns:
        x_reshaped: [days, 24, 5]
        y_reshaped: [days * 24]
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    # Reshape to daily batches
    x = x[:((x.size(0) // 24) * 24)].reshape((x.size(0) // 24, 24, 5))
    y = y[:((y.size(0) // 24) * 24)]
    
    # For 2h or 3h prediction, shift labels
    if predict_hours > 1:
        # Shift y by (predict_hours - 1) steps
        shift = predict_hours - 1
        y = y[shift:]
        x = x[:len(y) // 24]
        y = y[:len(x) * 24]
    
    return x, y


def split_train_test(x, y, train_ratio=0.75):
    """
    Split data into train and test sets (time series split)
    Args:
        x: [days, 24, 5]
        y: [days * 24]
        train_ratio: ratio of training data
    Returns:
        train_x, train_y, test_x, test_y
    """
    l = int(train_ratio * len(x))
    train_x = x[:l]
    train_y = y[:l*24]
    test_x = x[l:]
    test_y = y[l*24:]
    
    return train_x, train_y, test_x, test_y


class GRU_Model(nn.Module):
    """Wrapper for standard GRU model"""
    def __init__(self):
        super(GRU_Model, self).__init__()
        self.gru1 = standard_GRU(1, HIDDEN_DIM)
        self.gru2 = standard_GRU(HIDDEN_DIM, HIDDEN_DIM)
        self.fc = nn.Linear(HIDDEN_DIM, 1)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, input):
        # input: [batch_size, 24, 5] (weather + parking count)
        x_input = input[:, :, -1].unsqueeze(2)  # [batch_size, 24, 1]
        
        h1 = self.gru1(x_input)
        h2 = self.gru2(h1)
        out = h2.contiguous().view(-1, HIDDEN_DIM)
        out = self.fc(out).view(-1)
        return out


class PewGRU_Model(nn.Module):
    """Wrapper for PewGRU model"""
    def __init__(self):
        super(PewGRU_Model, self).__init__()
        self.gru1 = pew_GRU(1, HIDDEN_DIM, 4)
        self.gru2 = pew_GRU(HIDDEN_DIM, HIDDEN_DIM, 4)
        self.fc = nn.Linear(HIDDEN_DIM, 1)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, input):
        x_weather = input[:, :, :-1]  # [batch_size, 24, 4]
        x_input = input[:, :, -1].unsqueeze(2)  # [batch_size, 24, 1]
        
        h1 = self.gru1(x_input, x_weather)
        h2 = self.gru2(h1, x_weather)
        out = h2.contiguous().view(-1, HIDDEN_DIM)
        out = self.fc(out).view(-1)
        return out


class PewLSTM_Model(nn.Module):
    """Wrapper for PewLSTM model"""
    def __init__(self):
        super(PewLSTM_Model, self).__init__()
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


def train_model(model, train_x, train_y, epochs=500, lr=1e-2, 
                model_name='model', version='v1', checkpoint_dir=None,
                save_interval=50):
    """
    Train model with progress bar and checkpoint saving
    Args:
        model: PyTorch model
        train_x: training input
        train_y: training labels
        epochs: number of epochs
        lr: learning rate
        model_name: name for saving checkpoints
        version: version tag
        checkpoint_dir: directory for checkpoints
        save_interval: save checkpoint every N epochs
    Returns:
        trained model
    """
    if checkpoint_dir is None:
        checkpoint_dir = './checkpoints'
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Progress bar
    progress_bar = tqdm(range(epochs), desc=f'Training {model_name}')
    
    for epoch in progress_bar:
        model.train()
        out = model(train_x)
        loss = loss_fn(out, train_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.5f}'})
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'{checkpoint_dir}/{model_name}_{version}_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)
    
    # Save final model
    final_path = f'{checkpoint_dir}/{model_name}_{version}_final.pth'
    torch.save(model.state_dict(), final_path)
    
    return model


def resume_training(model, model_name, version, checkpoint_dir='./checkpoints'):
    """
    Resume training from latest checkpoint
    Returns:
        model, start_epoch
    """
    pattern = f'{checkpoint_dir}/{model_name}_{version}_epoch*.pth'
    checkpoint_files = sorted(glob.glob(pattern))
    
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ Resumed {model_name} from epoch {start_epoch}")
        return model, start_epoch
    
    return model, 0


def evaluate_model(model, test_x, test_y, scaler):
    """
    Evaluate model and calculate Accuracy and RMSE
    Args:
        model: trained model
        test_x: test input
        test_y: test labels
        scaler: MinMaxScaler for inverse transform
    Returns:
        accuracy, rmse
    """
    model.eval()
    with torch.no_grad():
        pred_test = model(test_x).cpu().detach().numpy().reshape(-1, 1)
    
    test_y_numpy = test_y.reshape(-1, 1).cpu().numpy()
    
    # Inverse transform for RMSE
    atest_y = scaler.inverse_transform(test_y_numpy)
    apred_test = scaler.inverse_transform(pred_test)
    
    # Calculate Accuracy (using normalized values)
    p = 0.0
    k = 0
    for i in range(len(test_y_numpy) - 1):
        k += 1
        if test_y_numpy[i] != 0:
            p += abs(test_y_numpy[i] - pred_test[i+1]) / test_y_numpy[i]
        else:
            p += abs(test_y_numpy[i] - pred_test[i+1])
    
    accuracy = (1 - p/k) * 100 if k > 0 else 0
    
    # Calculate RMSE (using denormalized values)
    r = 0
    for i in range(len(test_y_numpy) - 1):
        r += (atest_y[i] - apred_test[i+1])**2
    
    rmse = sqrt(r/k) if k > 0 else 0
    
    return accuracy[0], rmse


def run_mini_experiments(park_indices=range(10), predict_hours=1, 
                        task='departure', version='v1', epochs=500,
                        use_pretrained_pewlstm=True, output_dir='.'):
    """
    Run mini experiments: P1-P10, 1h, departure only
    Args:
        park_indices: list of parking lot indices (0-9 for P1-P10)
        predict_hours: prediction horizon (1, 2, or 3)
        task: 'departure' or 'arrival'
        version: version tag for this experiment
        epochs: number of training epochs
        use_pretrained_pewlstm: whether to use pretrained PewLSTM model
    Returns:
        DataFrame with results
    """
    results = []
    
    for park_idx in park_indices:
        print(f"\n{'='*60}")
        print(f"  Testing Park P{park_idx+1} | {predict_hours}h | {task}")
        print(f"{'='*60}")
        
        # Load data
        if task == 'departure':
            x, y, s = pGetAllData(park_idx)
        else:
            x, y, s = p2GetAllData(park_idx)
        
        x = x.astype('float32')
        y = y.astype('float32')
        
        # Prepare data
        x, y = prepare_data(x, y, predict_hours)
        train_x, train_y, test_x, test_y = split_train_test(x, y)
        
        print(f"  Data: {len(x)} days ({len(train_x)} train, {len(test_x)} test)")
        
        # Test PewLSTM
        if use_pretrained_pewlstm and predict_hours == 1 and park_idx in [0,1,2,3,4,5,6,7,8,9]:
            print(f"\n  [1/3] PewLSTM (using pretrained model)")
            pewlstm_model = PewLSTM_Model()
            # Try to load from output_dir first, then current dir
            model_path = os.path.join(output_dir, 'model_P1_1h.pth')
            if not os.path.exists(model_path):
                 model_path = 'model_P1_1h.pth'
            
            if os.path.exists(model_path):
                pewlstm_model.load_state_dict(torch.load(model_path))
            else:
                print(f"  Warning: Pretrained model not found at {model_path}, initializing new.")
        else:
            print(f"\n  [1/3] PewLSTM (training from scratch)")
            pewlstm_model = PewLSTM_Model()
            pewlstm_model = train_model(pewlstm_model, train_x, train_y, epochs=epochs,
                                       model_name=f'PewLSTM_P{park_idx+1}', version=version,
                                       checkpoint_dir=os.path.join(output_dir, 'checkpoints'))
        
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
        
        # Test GRU
        print(f"\n  [2/3] GRU (training)")
        gru_model = GRU_Model()
        gru_model = train_model(gru_model, train_x, train_y, epochs=epochs,
                               model_name=f'GRU_P{park_idx+1}', version=version,
                               checkpoint_dir=os.path.join(output_dir, 'checkpoints'))
        
        acc, rmse = evaluate_model(gru_model, test_x, test_y, s)
        results.append({
            'Park': f'P{park_idx+1}',
            'Model': 'GRU',
            'Hours': f'{predict_hours}h',
            'Task': task,
            'Accuracy': round(acc, 2),
            'RMSE': round(rmse, 2)
        })
        print(f"  ✓ GRU - Accuracy: {acc:.2f}%, RMSE: {rmse:.2f}")
        
        # Test PewGRU
        print(f"\n  [3/3] PewGRU (training)")
        pewgru_model = PewGRU_Model()
        pewgru_model = train_model(pewgru_model, train_x, train_y, epochs=epochs,
                                  model_name=f'PewGRU_P{park_idx+1}', version=version,
                                  checkpoint_dir=os.path.join(output_dir, 'checkpoints'))
        
        acc, rmse = evaluate_model(pewgru_model, test_x, test_y, s)
        results.append({
            'Park': f'P{park_idx+1}',
            'Model': 'PewGRU',
            'Hours': f'{predict_hours}h',
            'Task': task,
            'Accuracy': round(acc, 2),
            'RMSE': round(rmse, 2)
        })
        print(f"  ✓ PewGRU - Accuracy: {acc:.2f}%, RMSE: {rmse:.2f}")
    
    # Save results
    df = pd.DataFrame(results)
    result_path = os.path.join(output_dir, f'results_{version}.csv')
    df.to_csv(result_path, index=False)
    print(f"\n{'='*60}")
    print(f"  Results saved to: {result_path}")
    print(f"{'='*60}\n")
    
    return df


def run_full_experiments(park_indices=range(10), 
                        predict_hours_list=[1, 2, 3],
                        tasks=['departure', 'arrival'],
                        version='full_v1', epochs=500, output_dir='.'):
    """
    Run full experiments with all combinations
    """
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
                use_pretrained_pewlstm=False,
                output_dir=output_dir
            )
            all_results.append(df)
    
    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)
    final_path = os.path.join(output_dir, f'results_{version}_complete.csv')
    final_df.to_csv(final_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE!")
    print(f"  Final results saved to: {final_path}")
    print(f"{'='*60}\n")
    
    return final_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overall Testing System')
    parser.add_argument('--mini', action='store_true', help='Run mini version (P1-P10, 1h, departure)')
    parser.add_argument('--full', action='store_true', help='Run full version (all combinations)')
    parser.add_argument('--version', type=str, default='v1', help='Version tag')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--parks', type=str, default='all', help='Park indices (e.g., "0,1,2" or "all")')
    parser.add_argument('--hours', type=int, default=1, help='Prediction hours (1, 2, or 3)')
    parser.add_argument('--task', type=str, default='departure', choices=['departure', 'arrival'])
    
    args = parser.parse_args()
    
    # Parse park indices
    if args.parks == 'all':
        park_indices = range(10)
    else:
        park_indices = [int(x) for x in args.parks.split(',')]
    
    if args.mini:
        print("Running MINI experiments...")
        run_mini_experiments(
            park_indices=park_indices,
            predict_hours=args.hours,
            task=args.task,
            version=args.version,
            epochs=args.epochs,
            output_dir=script_dir
        )
    elif args.full:
        print("Running FULL experiments...")
        run_full_experiments(
            park_indices=park_indices,
            version=args.version,
            epochs=args.epochs,
            output_dir=script_dir
        )
    else:
        print("Please specify --mini or --full")
        print("Example: python overall.py --mini --version v1 --epochs 500")
