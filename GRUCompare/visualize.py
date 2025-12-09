"""
Visualization Tool for Model Comparison
Supports:
- Bar chart comparison for 3 models (PewLSTM, GRU, PewGRU)
- Multiple prediction horizons (1h, 2h, 3h)
- Multiple parking lots (P1-P10)
- Load results from CSV
- Handle missing values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


def load_results(csv_path):
    """
    Load results from CSV file
    Args:
        csv_path: path to CSV file
    Returns:
        DataFrame with results
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} results from {csv_path}")
    return df


def fill_missing_values(df):
    """
    Fill missing combinations with NaN
    Args:
        df: DataFrame with results
    Returns:
        DataFrame with filled missing values
    """
    # Get all unique values
    parks = sorted(df['Park'].unique())
    models = sorted(df['Model'].unique())
    hours = sorted(df['Hours'].unique())
    tasks = sorted(df['Task'].unique()) if 'Task' in df.columns else ['departure']
    
    # Create complete index
    from itertools import product
    complete_index = pd.DataFrame(
        list(product(parks, models, hours, tasks)),
        columns=['Park', 'Model', 'Hours', 'Task']
    )
    
    # Merge with original data
    df_complete = complete_index.merge(df, on=['Park', 'Model', 'Hours', 'Task'], how='left')
    
    missing_count = df_complete['Accuracy'].isna().sum()
    if missing_count > 0:
        print(f"  ⚠ Found {missing_count} missing combinations (filled with NaN)")
    
    return df_complete


def plot_comparison(df, metric='Accuracy', predict_hours='1h', task='departure',
                   parks=None, save_path=None, show=True):
    """
    Plot bar chart comparison for specified configuration
    Args:
        df: DataFrame with results
        metric: 'Accuracy' or 'RMSE'
        predict_hours: '1h', '2h', or '3h'
        task: 'departure' or 'arrival'
        parks: list of park names (e.g., ['P1', 'P2', 'P3'])
        save_path: path to save figure
        show: whether to display the plot
    """
    if parks is None:
        parks = [f'P{i+1}' for i in range(10)]
    
    # Filter data
    filtered = df[(df['Hours'] == predict_hours) & (df['Task'] == task)]
    filtered = filtered[filtered['Park'].isin(parks)]
    
    if len(filtered) == 0:
        print(f"⚠ No data found for {predict_hours} {task}")
        return
    
    # Prepare data
    models = ['PewLSTM', 'GRU', 'PewGRU']
    x = np.arange(len(parks))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
    
    for i, model in enumerate(models):
        values = []
        for park in parks:
            val = filtered[(filtered['Park'] == park) & (filtered['Model'] == model)][metric]
            if len(val) > 0 and not pd.isna(val.values[0]):
                values.append(val.values[0])
            else:
                values.append(0)  # Use 0 for missing values
        
        bars = ax.bar(x + i*width, values, width, label=model, color=colors[i], alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if not missing
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Parking Lot', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Comparison - {predict_hours} {task.capitalize()} Prediction', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(parks)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_metrics(df, parks=None, hours_list=['1h', '2h', '3h'], 
                      task='departure', save_path='multi_comparison.png'):
    """
    Plot multiple metrics (Accuracy + RMSE) for different prediction horizons
    Args:
        df: DataFrame with results
        parks: list of park names
        hours_list: list of prediction horizons
        task: 'departure' or 'arrival'
        save_path: path to save figure
    """
    if parks is None:
        parks = [f'P{i+1}' for i in range(10)]
    
    fig, axes = plt.subplots(2, len(hours_list), figsize=(18, 10))
    
    for col, hours in enumerate(hours_list):
        # Accuracy subplot
        ax_acc = axes[0, col] if len(hours_list) > 1 else axes[0]
        plot_single_metric(df, ax_acc, metric='Accuracy', predict_hours=hours,
                          task=task, parks=parks, title=f'{hours} Prediction')
        
        # RMSE subplot
        ax_rmse = axes[1, col] if len(hours_list) > 1 else axes[1]
        plot_single_metric(df, ax_rmse, metric='RMSE', predict_hours=hours,
                          task=task, parks=parks, title=f'{hours} Prediction')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved multi-metric plot to {save_path}")
    plt.show()


def plot_single_metric(df, ax, metric, predict_hours, task, parks, title=''):
    """Helper function to plot single metric on given axis"""
    filtered = df[(df['Hours'] == predict_hours) & (df['Task'] == task)]
    filtered = filtered[filtered['Park'].isin(parks)]
    
    models = ['PewLSTM', 'GRU', 'PewGRU']
    x = np.arange(len(parks))
    width = 0.25
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, model in enumerate(models):
        values = []
        for park in parks:
            val = filtered[(filtered['Park'] == park) & (filtered['Model'] == model)][metric]
            values.append(val.values[0] if len(val) > 0 and not pd.isna(val.values[0]) else 0)
        
        ax.bar(x + i*width, values, width, label=model, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Parking Lot')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} - {title}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(parks, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def plot_summary_table(df, save_path='summary_table.png'):
    """
    Create a summary table showing average metrics per model
    """
    summary = df.groupby(['Model', 'Hours', 'Task']).agg({
        'Accuracy': 'mean',
        'RMSE': 'mean'
    }).round(2).reset_index()
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(summary.to_string(index=False))
    print("="*60 + "\n")
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary.values, colLabels=summary.columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(summary.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Average Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary table to {save_path}")
    plt.show()


def plot_heatmap(df, metric='Accuracy', save_path='heatmap.png'):
    """
    Create a heatmap showing metric values across parks and models
    """
    # Pivot data for heatmap
    pivot_data = df.pivot_table(
        values=metric,
        index='Park',
        columns='Model',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlGnBu', 
                cbar_kws={'label': metric}, linewidths=0.5)
    plt.title(f'{metric} Heatmap - All Parks & Models', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Parking Lot', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to {save_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization Tool')
    parser.add_argument('--csv', type=str, default='results_v1.csv', 
                       help='Path to CSV file with results')
    parser.add_argument('--metric', type=str, default='Accuracy', 
                       choices=['Accuracy', 'RMSE'],
                       help='Metric to visualize')
    parser.add_argument('--hours', type=str, default='1h', 
                       help='Prediction hours (1h, 2h, 3h)')
    parser.add_argument('--task', type=str, default='departure',
                       choices=['departure', 'arrival'],
                       help='Task type')
    parser.add_argument('--parks', type=str, default='all',
                       help='Park indices (e.g., "P1,P2,P3" or "all")')
    parser.add_argument('--fill-missing', action='store_true',
                       help='Fill missing values with NaN')
    parser.add_argument('--multi', action='store_true',
                       help='Plot multiple metrics and horizons')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary table')
    parser.add_argument('--heatmap', action='store_true',
                       help='Plot heatmap')
    parser.add_argument('--output', type=str, default='comparison.png',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Load data
    df = load_results(args.csv)
    
    if args.fill_missing:
        df = fill_missing_values(df)
    
    # Parse parks
    if args.parks == 'all':
        parks = [f'P{i+1}' for i in range(10)]
    else:
        parks = args.parks.split(',')
    
    # Generate plots
    if args.summary:
        plot_summary_table(df, save_path='summary_table.png')
    
    if args.heatmap:
        plot_heatmap(df, metric=args.metric, save_path='heatmap.png')
    
    if args.multi:
        hours_list = ['1h', '2h', '3h'] if '2h' in df['Hours'].values else ['1h']
        plot_multi_metrics(df, parks=parks, hours_list=hours_list, 
                          task=args.task, save_path=args.output)
    else:
        plot_comparison(df, metric=args.metric, predict_hours=args.hours,
                       task=args.task, parks=parks, save_path=args.output)
    
    print("\n✓ Visualization complete!")
