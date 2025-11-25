import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from math import sqrt
import time
import datetime
import os
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from tqdm import tqdm
import argparse
import joblib

# Import models
from comparison_models import SimpleLSTM, RandomForestModel, AblationPewLSTM

# --- Data Processing Functions (Copied from main.py to avoid import side-effects) ---
HIDDEN_DIM = 1
SEQ_SIZE = 24
record_path = './data/record/'
weather_path = './data/weather/'
park_table_id = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']
park_weather_idx = [0,0,1,1,1,2,2,2,2,2]
weather_name = ['Ningbo','Ningbo Yinzhou','Changsha']

def read_park_table(index, debug = False):
    park_table_path = record_path + park_table_id[index] + '.csv'
    park_book = pd.read_csv(park_table_path,encoding='ISO-8859-1')
    return park_book

def read_weather_table(index, debug = False):
    weather_table_path = weather_path + str(index) + '.csv'
    weather_book = pd.read_csv(weather_table_path,encoding='ISO-8859-1')
    return weather_book

def process_weather(data, debug= False):
    output = []
    start_h = data['DAY'][0]
    start_h = int(time.mktime(time.strptime(start_h,"%Y/%m/%d %H:%M")) // (60*60))
    output.append(start_h)
    
    for i in range(5):
        output.append([])
    output.append({})
    for i in range(len(data['HOUR'])):
        output[1].append(data['TEM'][i])
        output[2].append(data['RHU'][i])
        output[3].append(data['WIN_S'][i])
        output[4].append(data['PRE_1h'][i])
        output[5].append(time.strptime(data['DAY'][i],"%Y/%m/%d %H:%M").tm_wday)
        output[6][int(time.mktime(time.strptime(data['DAY'][i],"%Y/%m/%d %H:%M")) // (60*60))] = i
    return output

def invalid(w_list,idx):
    if w_list[1][idx] > 999: return True
    if w_list[2][idx] > 999: return True
    if w_list[3][idx] > 999: return True
    if w_list[4][idx] > 999: return True
    return False

def is_valid(w_list,idx):
    flag = [1,1,1,1]
    for i in range(1,5):
        if w_list[i][idx] > 999:
            flag[i-1] = 0
    return flag

def valid_weather(w_list,idx): 
    flag = is_valid(w_list,idx)
    temp = [0,0,0,0]
    d = 0
    for i in range(1,5):
        if flag[i-1] == 0:
            d = idx - 1
            while (is_valid(w_list,d)[i-1] == 0):
                d -= 1
            upvalue = w_list[i][d]
            d = idx + 1
            while (is_valid(w_list,d)[i-1] == 0):
                d += 1
            downvalue = w_list[i][d]
            temp[i-1] = 0.5 * (upvalue + downvalue)
        else:
            temp[i-1] = w_list[i][d]
    return temp

def calc_park_cnt_from_dict(p_dict, debug = False):
    park_cnt = []
    if not p_dict: return park_cnt
    st = min(p_dict.keys())
    ed = max(p_dict.keys())
    for i in range(st,ed+1):
        if i in p_dict:
            park_cnt.append(p_dict[i]['cnt'])
        else:
            park_cnt.append(0)
    return park_cnt

def gen_series(park_cnt, weather_rec, start_h, end_h, debug=False):
    tt = []
    for i in range(len(park_cnt)):
        tt.append(start_h + i)
    temp = []
    for i in range(5):
        temp.append([])
    for i in range(len(park_cnt)):
        if tt[i] in weather_rec[6]:
            idx = weather_rec[6][tt[i]]
            temp[0].append(park_cnt[i])
            if invalid(weather_rec,idx):
                vld = valid_weather(weather_rec,idx)
                temp[1].append(vld[0])
                temp[2].append(vld[1])
                temp[3].append(vld[2])
                temp[4].append(vld[3])
            else:
                temp[1].append(weather_rec[1][idx])
                temp[2].append(weather_rec[2][idx])
                temp[3].append(weather_rec[3][idx])
                temp[4].append(weather_rec[4][idx])
    
    park_cnt = pd.Series(temp[0], name='cnt')
    tem = pd.Series(temp[1], name='tem')
    rhu = pd.Series(temp[2], name='rhu')
    winds = pd.Series(temp[3], name='wind_s')
    pre_1h = pd.Series(temp[4],name='pre_ih')
    output = pd.concat([tem,rhu,winds,pre_1h,park_cnt], axis=1)
    return output

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Departure
def ptrans_record_to_count(data, debug = False):
    p_dict = {}
    for stime,etime in zip(data['Lockdown Time'],data['Lockup Time']):
        start_tss = time.strptime(stime, "%Y/%m/%d %H:%M")
        end_tss = time.strptime(etime, "%Y/%m/%d %H:%M")
        start_tsp = int(time.mktime(start_tss))
        end_tsp = int(time.mktime(end_tss))
        if end_tsp - start_tsp <= 5*60:
            continue
        end_hour = int(end_tsp//(60*60))
        if end_hour not in p_dict:
            p_dict[end_hour] = {}
            p_dict[end_hour]['cnt'] = 1
        else:
            p_dict[end_hour]['cnt'] += 1
    return p_dict

# Arrival
def p2trans_record_to_count(data, debug = False):
    p_dict = {}
    for stime,etime in zip(data['Lockdown Time'],data['Lockup Time']):
        start_tss = time.strptime(stime, "%Y/%m/%d %H:%M")
        end_tss = time.strptime(etime, "%Y/%m/%d %H:%M")
        start_tsp = int(time.mktime(start_tss))
        end_tsp = int(time.mktime(end_tss))
        if end_tsp - start_tsp <= 5*60:
            continue
        start_hour = int(start_tsp//(60*60))
        if start_hour not in p_dict:
            p_dict[start_hour] = {}
            p_dict[start_hour]['cnt'] = 1
        else:
            p_dict[start_hour]['cnt'] += 1
    return p_dict

def pGetAllData(index):
    park_book = read_park_table(index)
    weather_book = read_weather_table(park_weather_idx[index])
    p_dic = ptrans_record_to_count(park_book)
    start_h = min(p_dic.keys())
    end_h = max(p_dic.keys())
    park_cnt = calc_park_cnt_from_dict(p_dic)
    weather_rec = process_weather(weather_book)
    p_series = gen_series(park_cnt, weather_rec, start_h, end_h,debug=True)
    
    # Fill NA logic from main.py
    p = gen_series(park_cnt, weather_rec, start_h, end_h,debug=True)
    p.fillna(value = 0,inplace=True)
    values = [0,0,0,0]
    for k in range(len(p_series)):
        values[0] += p['tem'][k]
        values[1] += p['rhu'][k]
        values[2] += p['wind_s'][k]
        values[3] += p['pre_ih'][k]
    for i in range(4):
        values[i] /= (len(p_series))
    p_series['tem'].fillna(value=values[0],inplace=True)
    p_series['rhu'].fillna(value=values[1],inplace=True)
    p_series['wind_s'].fillna(value=values[2],inplace=True)
    p_series['pre_ih'].fillna(value=values[3],inplace=True)

    p_series = p_series.astype('float32')
    scaler = MinMaxScaler(feature_range=(0,1))
    sclaed = scaler.fit_transform(p_series)
    reframed = series_to_supervised(sclaed, 1, 1)
    reframed.drop(reframed.columns[[5,6,7,8]], axis=1, inplace=True)

    s = MinMaxScaler(feature_range=(0,1))
    m = p_series
    m = series_to_supervised(m, 1, 1)
    m.drop(m.columns[[5,6,7,8]], axis=1, inplace=True)
    m2 = m.values[:,-1]
    m2 = m2.reshape(-1,1)
    m22 = s.fit_transform(m2)
    return (reframed.values[:,:-1],reframed.values[:,-1],s)

def p2GetAllData(index):
    park_book = read_park_table(index)
    weather_book = read_weather_table(park_weather_idx[index])
    p_dic = p2trans_record_to_count(park_book)
    start_h = min(p_dic.keys())
    end_h = max(p_dic.keys())
    park_cnt = calc_park_cnt_from_dict(p_dic)
    weather_rec = process_weather(weather_book)
    p_series = gen_series(park_cnt, weather_rec, start_h, end_h,debug=True)
    
    # Fill NA logic (assuming same as pGetAllData for robustness)
    p = gen_series(park_cnt, weather_rec, start_h, end_h,debug=True)
    p.fillna(value = 0,inplace=True)
    values = [0,0,0,0]
    for k in range(len(p_series)):
        values[0] += p['tem'][k]
        values[1] += p['rhu'][k]
        values[2] += p['wind_s'][k]
        values[3] += p['pre_ih'][k]
    for i in range(4):
        values[i] /= (len(p_series))
    p_series['tem'].fillna(value=values[0],inplace=True)
    p_series['rhu'].fillna(value=values[1],inplace=True)
    p_series['wind_s'].fillna(value=values[2],inplace=True)
    p_series['pre_ih'].fillna(value=values[3],inplace=True)

    p_series = p_series.astype('float32')
    scaler = MinMaxScaler(feature_range=(0,1))
    sclaed = scaler.fit_transform(p_series)
    reframed = series_to_supervised(sclaed, 1, 1)
    reframed.drop(reframed.columns[[5,6,7,8]], axis=1, inplace=True)

    s = MinMaxScaler(feature_range=(0,1))
    m = p_series
    m = series_to_supervised(m, 1, 1)
    m.drop(m.columns[[5,6,7,8]], axis=1, inplace=True)
    m2 = m.values[:,-1]
    m2 = m2.reshape(-1,1)
    m22 = s.fit_transform(m2)
    return (reframed.values[:,:-1],reframed.values[:,-1],s)

# --- Experiment Logic ---

def train_and_evaluate(model_name, model, train_x, train_y, test_x, test_y, scaler, epochs=500, checkpoint_path=None, evaluate_only=False):
    
    # Check for checkpoint
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        if isinstance(model, RandomForestModel):
            try:
                model.model = joblib.load(checkpoint_path)
            except Exception as e:
                print(f"Failed to load RF model: {e}")
        else:
            checkpoint = torch.load(checkpoint_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
    
    if not evaluate_only:
        if isinstance(model, RandomForestModel):
            print(f"Training {model_name}...")
            model.fit(train_x, train_y)
            # Save RF model
            if checkpoint_path:
                joblib.dump(model.model, checkpoint_path)
        else:
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            
            if checkpoint_path and os.path.exists(checkpoint_path) and not isinstance(model, RandomForestModel):
                checkpoint = torch.load(checkpoint_path)
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if start_epoch < epochs:
                print(f"Training {model_name} from epoch {start_epoch+1} to {epochs}...")
                pbar = tqdm(range(start_epoch, epochs), desc=f"Training {model_name}")
                for i in pbar:
                    model.train()
                    # Convert to tensor if not already
                    tx = torch.from_numpy(train_x) if isinstance(train_x, np.ndarray) else train_x
                    ty = torch.from_numpy(train_y) if isinstance(train_y, np.ndarray) else train_y
                    
                    out = model(tx)
                    loss = loss_function(out, ty)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    pbar.set_postfix({'loss': f"{loss.item():.5f}"})
                    
                    # Save checkpoint
                    if checkpoint_path:
                        torch.save({
                            'epoch': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                        }, checkpoint_path)
        
    if isinstance(model, RandomForestModel):
         pred_test = model.predict(test_x).reshape(-1, 1)
    else:
        model.eval()
        tx_test = torch.from_numpy(test_x) if isinstance(test_x, np.ndarray) else test_x
        pred_test = model(tx_test).cpu().detach().numpy().reshape(-1,1)

    # Evaluation (Same as main.py)
    p = 0.0
    r = 0
    k = 0
    test_y_numpy = test_y.reshape(-1,1)
    if isinstance(test_y, torch.Tensor):
        test_y_numpy = test_y.cpu().numpy().reshape(-1,1)
        
    atest_y = scaler.inverse_transform(test_y_numpy)
    apred_test = scaler.inverse_transform(pred_test)
    
    for i in range(len(test_y_numpy)-1):
        k+=1
        if test_y_numpy[i] != 0:
            p = p + (abs(test_y_numpy[i]-pred_test[i+1])/test_y_numpy[i])
        else:
            p = p + abs(test_y_numpy[i]-pred_test[i+1])
        r += (atest_y[i]-apred_test[i+1])**2
    
    accuracy = (1-p/k)*100
    rmse = sqrt(r/k)
    
    # Handle accuracy being an array sometimes
    if isinstance(accuracy, np.ndarray):
        accuracy = accuracy[0]
        
    return accuracy, rmse

import argparse

def run_experiment():
    parser = argparse.ArgumentParser(description='Run parking prediction experiments.')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['SimpleLSTM', 'RandomForest', 'PewLSTM', 'PewLSTM_no_periodic', 'PewLSTM_no_weather', 'all'],
                        help='Model to run')
    parser.add_argument('--park', type=str, default='all', help='Park ID (e.g., P1) or "all"')
    parser.add_argument('--type', type=str, default='all', choices=['Departure', 'Arrival', 'all'], help='Prediction type')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--version', type=str, default=None, help='Experiment version name (default: timestamp)')
    parser.add_argument('--evaluate_only', action='store_true', help='Skip training and only evaluate loaded models')
    args = parser.parse_args()

    if args.version is None:
        args.version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Experiment Version: {args.version}")

    results = []
    
    # Create checkpoints directory
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # Models to test
    all_models = [
        {'name': 'SimpleLSTM', 'type': 'SimpleLSTM'},
        {'name': 'RandomForest', 'type': 'RandomForest'},
        {'name': 'PewLSTM', 'type': 'PewLSTM', 'periodic': True, 'weather': True},
        {'name': 'PewLSTM_no_periodic', 'type': 'PewLSTM', 'periodic': False, 'weather': True},
        {'name': 'PewLSTM_no_weather', 'type': 'PewLSTM', 'periodic': True, 'weather': False},
    ]
    
    if args.model == 'all':
        models_config = all_models
    else:
        models_config = [m for m in all_models if m['name'] == args.model]

    # Datasets
    if args.park == 'all':
        parks_to_run = range(len(park_table_id))
    else:
        if args.park in park_table_id:
            parks_to_run = [park_table_id.index(args.park)]
        else:
            print(f"Invalid park ID: {args.park}")
            return

    for park_idx in parks_to_run:
        park_name = park_table_id[park_idx]
        print(f"\n=== Processing {park_name} ===")
        
        # Run Departure
        if args.type in ['Departure', 'all']:
            print("Loading Departure Data...")
            x, y, s = pGetAllData(park_idx)
            x = x.astype('float32')
            y = y.astype('float32')
            
            n_days = x.shape[0] // 24
            x_cut = x[:(n_days * 24)]
            y_cut = y[:(n_days * 24)]
            
            x_reshaped = x_cut.reshape((n_days, 24, 5))
            
            l = int(0.75 * n_days)
            train_x = x_reshaped[:l]
            train_y = y_cut[:l*24]
            test_x = x_reshaped[l:]
            test_y = y_cut[l*24:]
            
            for config in models_config:
                model_name = config['name']
                print(f"Running {model_name} on {park_name} (Departure)...")
                
                checkpoint_path = f"checkpoints/{args.version}_{park_name}_{model_name}_departure.pth"
                
                if config['type'] == 'SimpleLSTM':
                    model = SimpleLSTM(input_size=1, hidden_size=1)
                elif config['type'] == 'RandomForest':
                    model = RandomForestModel()
                elif config['type'] == 'PewLSTM':
                    model = AblationPewLSTM(hidden_dim=1, 
                                            use_periodic=config['periodic'], 
                                            use_weather=config['weather'])
                
                acc, rmse = train_and_evaluate(model_name, model, train_x, train_y, test_x, test_y, s, 
                                               epochs=args.epochs, checkpoint_path=checkpoint_path, evaluate_only=args.evaluate_only)
                
                print(f"Result: Accuracy={acc:.2f}%, RMSE={rmse:.2f}")
                results.append({
                    'version': args.version,
                    'park': park_name,
                    'type': 'Departure',
                    'model': model_name,
                    'accuracy': acc,
                    'rmse': rmse
                })

        # Run Arrival
        if args.type in ['Arrival', 'all']:
            print("Loading Arrival Data...")
            x2, y2, s2 = p2GetAllData(park_idx)
            x2 = x2.astype('float32')
            y2 = y2.astype('float32')
            
            n_days2 = x2.shape[0] // 24
            x2_cut = x2[:(n_days2 * 24)]
            y2_cut = y2[:(n_days2 * 24)]
            
            x2_reshaped = x2_cut.reshape((n_days2, 24, 5))
            
            l2 = int(0.75 * n_days2)
            train_x2 = x2_reshaped[:l2]
            train_y2 = y2_cut[:l2*24]
            test_x2 = x2_reshaped[l2:]
            test_y2 = y2_cut[l2*24:]
            
            for config in models_config:
                model_name = config['name']
                print(f"Running {model_name} on {park_name} (Arrival)...")
                
                checkpoint_path = f"checkpoints/{args.version}_{park_name}_{model_name}_arrival.pth"
                
                if config['type'] == 'SimpleLSTM':
                    model = SimpleLSTM(input_size=1, hidden_size=1)
                elif config['type'] == 'RandomForest':
                    model = RandomForestModel()
                elif config['type'] == 'PewLSTM':
                    model = AblationPewLSTM(hidden_dim=1, 
                                            use_periodic=config['periodic'], 
                                            use_weather=config['weather'])
                
                acc, rmse = train_and_evaluate(model_name, model, train_x2, train_y2, test_x2, test_y2, s2, 
                                               epochs=args.epochs, checkpoint_path=checkpoint_path, evaluate_only=args.evaluate_only)
                
                print(f"Result: Accuracy={acc:.2f}%, RMSE={rmse:.2f}")
                results.append({
                    'version': args.version,
                    'park': park_name,
                    'type': 'Arrival',
                    'model': model_name,
                    'accuracy': acc,
                    'rmse': rmse
                })

    # Save all results
    if results:
        df_res = pd.DataFrame(results)
        print("\n=== Final Results ===")
        print(df_res)
        # Append to existing file if it exists, or create new
        output_file = 'overall_results.csv'
        if os.path.exists(output_file):
            # Check if header exists
            try:
                existing_df = pd.read_csv(output_file)
                if 'version' not in existing_df.columns:
                    # If old file doesn't have version, we might want to add it or just append (pandas handles missing cols by adding NaNs usually, but let's be safe)
                    pass
            except:
                pass
            df_res.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        else:
            df_res.to_csv(output_file, index=False)


if __name__ == "__main__":
    run_experiment()
