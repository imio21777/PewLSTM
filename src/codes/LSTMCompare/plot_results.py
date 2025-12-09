import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def plot_comparison(csv_file, output_dir='compare', parks=None, hours=None):
    """
    Plot comparison bar charts from results CSV
    """
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    
    # Filter by parks if specified
    if parks:
        park_list = [f'P{p}' for p in parks.split(',')]
        df = df[df['Park'].isin(park_list)]
    
    # Filter by hours if specified
    if hours:
        hour_label = f'{hours}h'
        df = df[df['Hours'] == hour_label]
    
    # Get unique models and parks
    models = df['Model'].unique()
    parks = df['Park'].unique()
    
    # Sort parks naturally (P1, P2, ..., P10)
    parks = sorted(parks, key=lambda x: int(x[1:]))
    
    # Metrics to plot
    metrics = ['RMSE', 'Accuracy']
    
    for metric in metrics:
        plt.figure(figsize=(15, 8))
        
        # Set width of bar
        barWidth = 0.15
        
        # Set position of bar on X axis
        r = np.arange(len(parks))
        
        # Plot bars for each model
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            
            # Align data with parks order
            values = []
            for park in parks:
                val = model_data[model_data['Park'] == park][metric].values
                if len(val) > 0:
                    values.append(val[0])
                else:
                    values.append(0)
            
            plt.bar(r + i*barWidth, values, width=barWidth, edgecolor='white', label=model)
        
        # Add xticks on the middle of the group bars
        plt.xlabel('Parking Lot', fontweight='bold')
        plt.xticks([r + barWidth * (len(models)-1) / 2 for r in range(len(parks))], parks)
        
        plt.ylabel(metric, fontweight='bold')
        plt.title(f'Model Comparison - {metric}', fontweight='bold')
        plt.legend()
        
        # Save plot
        output_path = os.path.join(output_dir, f'comparison_{metric}.png')
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Model Comparison Results')
    parser.add_argument('--file', type=str, required=True, help='Path to results CSV file')
    parser.add_argument('--parks', type=str, help='Comma separated park indices (e.g. 1,2,3)')
    parser.add_argument('--hours', type=int, help='Prediction hours (1, 2, or 3)')
    
    args = parser.parse_args()
    
    plot_comparison(args.file, parks=args.parks, hours=args.hours)
