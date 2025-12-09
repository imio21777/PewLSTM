"""
Visualization Tool for 5 Model Comparison
支持5种模型对比可视化:
1. PewLSTM
2. SimpleLSTM  
3. RandomForest
4. PewLSTM w/o Periodic
5. PewLSTM w/o Weather
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['font.size'] = 10


def load_results(csv_path):
    """加载CSV结果"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} results from {csv_path}")
    return df


def fill_missing_values(df):
    """填充缺失值"""
    from itertools import product
    
    parks = sorted(df['Park'].unique())
    models = sorted(df[ 'Model'].unique())
    hours = sorted(df['Hours'].unique())
    tasks = sorted(df['Task'].unique()) if 'Task' in df.columns else ['departure']
    
    complete_index = pd.DataFrame(
        list(product(parks, models, hours, tasks)),
        columns=['Park', 'Model', 'Hours', 'Task']
    )
    
    df_complete = complete_index.merge(df, on=['Park', 'Model', 'Hours', 'Task'], how='left')
    
    missing_count = df_complete['Accuracy'].isna().sum()
    if missing_count > 0:
        print(f"  ⚠ Found {missing_count} missing combinations (filled with NaN)")
    
    return df_complete


def plot_comparison(df, metric='Accuracy', predict_hours='1h', task='departure',
                   parks=None, save_path=None, show=True):
    """
    绘制5种模型对比柱状图
    """
    if parks is None:
        parks = [f'P{i+1}' for i in range(10)]
    
    # 过滤数据
    filtered = df[(df['Hours'] == predict_hours) & (df['Task'] == task)]
    filtered = filtered[filtered['Park'].isin(parks)]
    
    if len(filtered) == 0:
        print(f"⚠ No data found for {predict_hours} {task}")
        return
    
    # 5种模型
    models = ['PewLSTM', 'SimpleLSTM', 'RandomForest', 
              'PewLSTM_w/o_Periodic', 'PewLSTM_w/o_Weather']
    model_labels = ['PewLSTM', 'SimpleLSTM', 'RandomForest',
                   'w/o Periodic', 'w/o Weather']
    
    x = np.arange(len(parks))
    width = 0.15  # 5个柱子
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    for i, (model, label) in enumerate(zip(models, model_labels)):
        values = []
        for park in parks:
            val = filtered[(filtered['Park'] == park) & (filtered['Model'] == model)][metric]
            if len(val) > 0 and not pd.isna(val.values[0]):
                values.append(val.values[0])
            else:
                values.append(0)
        
        bars = ax.bar(x + i*width, values, width, label=label, color=colors[i], alpha=0.85)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Parking Lot', fontsize=13, fontweight='bold')
    ax.set_ylabel(metric, fontsize=13, fontweight='bold')
    ax.set_title(f'{metric} Comparison - {predict_hours} {task.capitalize()} Prediction (5 Models)', 
                fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(parks)
    ax.legend(loc='best', frameon=True, shadow=True, ncol=5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_mini_comparison(df, save_path='mini_comparison.png'):
    """
    Mini版本可视化: 1h, departure, 所有模型, 所有停车场
    """
    parks = [f'P{i+1}' for i in range(10)]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Accuracy
    plot_single_metric(df, axes[0], metric='Accuracy', predict_hours='1h',
                      task='departure', parks=parks)
    
    # RMSE
    plot_single_metric(df, axes[1], metric='RMSE', predict_hours='1h',
                      task='departure', parks=parks)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved mini comparison to {save_path}")
    plt.show()


def plot_single_metric(df, ax, metric, predict_hours, task, parks):
    """在指定ax上绘制单个指标"""
    filtered = df[(df['Hours'] == predict_hours) & (df['Task'] == task)]
    filtered = filtered[filtered['Park'].isin(parks)]
    
    models = ['PewLSTM', 'SimpleLSTM', 'RandomForest',
              'PewLSTM_w/o_Periodic', 'PewLSTM_w/o_Weather']
    model_labels = ['PewLSTM', 'SimpleLSTM', 'RF', 'w/o Periodic', 'w/o Weather']
    
    x = np.arange(len(parks))
    width = 0.15
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    for i, (model, label) in enumerate(zip(models, model_labels)):
        values = []
        for park in parks:
            val = filtered[(filtered['Park'] == park) & (filtered['Model'] == model)][metric]
            values.append(val.values[0] if len(val) > 0 and not pd.isna(val.values[0]) else 0)
        
        ax.bar(x + i*width, values, width, label=label, color=colors[i], alpha=0.85)
    
    ax.set_xlabel('Parking Lot', fontweight='bold')
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_title(f'{metric} - {predict_hours} {task.capitalize()}', fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(parks)
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(axis='y', alpha=0.3)


def plot_multi_hours(df, parks=None, hours_list=['1h', '2h', '3h'],
                    task='departure', save_path='multi_hours.png'):
    """
    多时长对比 (1h/2h/3h)
    """
    if parks is None:
        parks = [f'P{i+1}' for i in range(10)]
    
    fig, axes = plt.subplots(2, len(hours_list), figsize=(18, 10))
    
    for col, hours in enumerate(hours_list):
        # Accuracy
        ax_acc = axes[0, col] if len(hours_list) > 1 else axes[0]
        plot_single_metric(df, ax_acc, metric='Accuracy', predict_hours=hours,
                          task=task, parks=parks)
        
        # RMSE
        ax_rmse = axes[1, col] if len(hours_list) > 1 else axes[1]
        plot_single_metric(df, ax_rmse, metric='RMSE', predict_hours=hours,
                          task=task, parks=parks)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved multi-hours comparison to {save_path}")
    plt.show()


def plot_summary_table(df, save_path='summary_table.png'):
    """创建摘要表格"""
    summary = df.groupby(['Model', 'Hours', 'Task']).agg({
        'Accuracy': 'mean',
        'RMSE': 'mean'
    }).round(2).reset_index()
    
    print("\n" + "="*70)
    print("SUMMARY TABLE (Average across all parks)")
    print("="*70)
    print(summary.to_string(index=False))
    print("="*70 + "\n")
    
    # 创建可视化表格
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary.values, colLabels=summary.columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 着色表头
    for i in range(len(summary.columns)):
        table[(0, i)].set_facecolor('#2ecc71')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Average Performance Summary (5 Models)', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary table to {save_path}")
    plt.show()


def plot_heatmap(df, metric='Accuracy', hours='1h', task='departure', 
                save_path='heatmap.png'):
    """创建热图"""
    filtered = df[(df['Hours'] == hours) & (df['Task'] == task)]
    
    pivot_data = filtered.pivot_table(
        values=metric,
        index='Park',
        columns='Model',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn',
                cbar_kws={'label': metric}, linewidths=0.5, vmin=0, vmax=100 if metric=='Accuracy' else None)
    plt.title(f'{metric} Heatmap - {hours} {task.capitalize()} (5 Models)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Parking Lot', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to {save_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='5 Model Visualization Tool')
    parser.add_argument('--csv', type=str, default='results_v1.csv',
                       help='CSV file path')
    parser.add_argument('--metric', type=str, default='Accuracy',
                       choices=['Accuracy', 'RMSE'],
                       help='Metric to visualize')
    parser.add_argument('--hours', type=str, default='1h',
                       help='Prediction hours (1h/2h/3h)')
    parser.add_argument('--task', type=str, default='departure',
                       choices=['departure', 'arrival'],
                       help='Task type')
    parser.add_argument('--parks', type=str, default='all',
                       help='Park list (e.g., "P1,P2,P3" or "all")')
    parser.add_argument('--fill-missing', action='store_true',
                       help='Fill missing values')
    parser.add_argument('--mini', action='store_true',
                       help='Mini version visualization (1h, departure, all parks)')
    parser.add_argument('--multi-hours', action='store_true',
                       help='Multi-hours comparison (1h/2h/3h)')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary table')
    parser.add_argument('--heatmap', action='store_true',
                       help='Plot heatmap')
    parser.add_argument('--output', type=str, default='comparison.png',
                       help='Output file name')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_results(args.csv)
    
    if args.fill_missing:
        df = fill_missing_values(df)
    
    # Parse parks
    if args.parks == 'all':
        parks = [f'P{i+1}' for i in range(10)]
    else:
        parks = args.parks.split(',')
    
    # 生成图表
    if args.mini:
        plot_mini_comparison(df, save_path=args.output)
    elif args.multi_hours:
        hours_list = ['1h', '2h', '3h'] if '2h' in df['Hours'].values else ['1h']
        plot_multi_hours(df, parks=parks, hours_list=hours_list,
                        task=args.task, save_path=args.output)
    elif args.summary:
        plot_summary_table(df, save_path=args.output)
    elif args.heatmap:
        plot_heatmap(df, metric=args.metric, hours=args.hours,
                    task=args.task, save_path=args.output)
    else:
        plot_comparison(df, metric=args.metric, predict_hours=args.hours,
                       task=args.task, parks=parks, save_path=args.output)
    
    print("\n✓ Visualization complete!")
