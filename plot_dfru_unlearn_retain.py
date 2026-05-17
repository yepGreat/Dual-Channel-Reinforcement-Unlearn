"""
DFRU Unlearn vs Retain Comparison Plot
DFRU方法在unlearn set和retain set上的性能对比柱状图
横坐标: Unlearn Epoch (5, 10, 15, 20, 25, 30, 35, 40)
纵坐标: 4个指标 - Average Steps, Rewards, Collision Rate (%), Perplexity
忽略reward_scale和poison_intensity，对每个epoch取所有组合的均值
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# Configuration
STEPS_LIST = [15, 20, 25, 30, 35, 40]
UNLEARN_EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40]
DFRU_XLSX_TEMPLATE = "./result/grid_world/DFRU_Model_standard_epoch_50/tem_dfru_max_steps_{step}/dfru_verification_filtered.xlsx"
DFRU_OUTPUT_TEMPLATE = "dfru_unlearn_retain_comparison_step_{step}.png"


def load_dfru_data(xlsx_file, unlearn_epochs):
    """Load DFRU verification results for both unlearn and retain sets."""
    if not os.path.exists(xlsx_file):
        print(f"Warning: File not found: {xlsx_file}")
        return None

    try:
        df = pd.read_excel(xlsx_file)
        print(f"Loaded DFRU: {len(df)} rows from {xlsx_file}")
    except Exception as e:
        print(f"Error reading {xlsx_file}: {e}")
        return None

    result = {
        'pretrain': {'unlearn': None, 'retain': None},
        'retrain': {'unlearn': None, 'retain': None},
        'epoch_data': {}
    }

    # Extract Pretrain baseline
    pretrain_rows = df[df['model_type'] == 'Pretrain']
    if len(pretrain_rows) > 0:
        row = pretrain_rows.iloc[0]
        for data_cat in ['unlearn', 'retain']:
            col_prefix = f'original_{data_cat}_'
            result['pretrain'][data_cat] = {
                'avg_steps': float(row[f'{col_prefix}avg_steps']),
                'avg_reward': float(row[f'{col_prefix}avg_reward']),
                'unlearn_ratio': float(row[f'{col_prefix}unlearn_ratio']),
                'avg_perplexity': float(row[f'{col_prefix}avg_perplexity'])
            }
        print(f"  Pretrain loaded")

    # Extract Retrain baseline
    retrain_rows = df[df['model_type'] == 'Retrain']
    if len(retrain_rows) > 0:
        row = retrain_rows.iloc[0]
        for data_cat in ['unlearn', 'retain']:
            col_prefix = f'original_{data_cat}_'
            result['retrain'][data_cat] = {
                'avg_steps': float(row[f'{col_prefix}avg_steps']),
                'avg_reward': float(row[f'{col_prefix}avg_reward']),
                'unlearn_ratio': float(row[f'{col_prefix}unlearn_ratio']),
                'avg_perplexity': float(row[f'{col_prefix}avg_perplexity'])
            }
        print(f"  Retrain loaded")

    # Extract DFRU data (aggregate by epoch, ignoring reward_scale and poison_intensity)
    dfru_rows = df[df['model_type'] == 'DFRU']
    if len(dfru_rows) > 0:
        for epoch in unlearn_epochs:
            epoch_rows = dfru_rows[dfru_rows['unlearn_epoch'] == epoch]
            result['epoch_data'][epoch] = {'unlearn': None, 'retain': None}

            if len(epoch_rows) > 0:
                for data_cat in ['unlearn', 'retain']:
                    col_prefix = f'original_{data_cat}_'
                    result['epoch_data'][epoch][data_cat] = {
                        'avg_steps': epoch_rows[f'{col_prefix}avg_steps'].astype(float).mean(),
                        'avg_reward': epoch_rows[f'{col_prefix}avg_reward'].astype(float).mean(),
                        'unlearn_ratio': epoch_rows[f'{col_prefix}unlearn_ratio'].astype(float).mean(),
                        'avg_perplexity': epoch_rows[f'{col_prefix}avg_perplexity'].astype(float).mean(),
                        'count': len(epoch_rows)
                    }
            else:
                for data_cat in ['unlearn', 'retain']:
                    result['epoch_data'][epoch][data_cat] = {
                        'avg_steps': np.nan, 'avg_reward': np.nan,
                        'unlearn_ratio': np.nan, 'avg_perplexity': np.nan, 'count': 0
                    }

        epochs_with_data = [e for e in unlearn_epochs
                           if result['epoch_data'][e]['unlearn'] and result['epoch_data'][e]['unlearn']['count'] > 0]
        print(f"  DFRU epochs with data: {epochs_with_data}")

    return result


def plot_dfru_comparison(data, unlearn_epochs, output_file, step):
    """Create 2x2 subplot comparing unlearn set and retain set for DFRU."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'DFRU Performance: Unlearn Set vs Retain Set',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.subplots_adjust(hspace=0.3, wspace=0.25, top=0.88)

    colors = {
        'unlearn': (172/255, 200/255, 232/255),  # Light Blue
        'retain': (237/255, 182/255, 145/255),   # Light Orange
        'Pretrain': '#27AE60',
        'Retrain': '#8E44AD'
    }

    bar_width = 0.35
    x = np.arange(len(unlearn_epochs))

    metrics = [
        ('avg_steps', 'Average Steps', axes[0, 0]),
        ('avg_reward', 'Rewards', axes[0, 1]),
        ('unlearn_ratio', 'Target Obstacle Collision Rate (%)', axes[1, 0]),
        ('avg_perplexity', 'Near Obstacle Perplexity', axes[1, 1])
    ]

    for metric_key, ylabel, ax in metrics:
        # Get values for both sets
        unlearn_values = [data['epoch_data'].get(e, {}).get('unlearn', {}).get(metric_key, np.nan)
                         for e in unlearn_epochs]
        retain_values = [data['epoch_data'].get(e, {}).get('retain', {}).get(metric_key, np.nan)
                        for e in unlearn_epochs]

        # Plot bars
        ax.bar(x - bar_width/2, unlearn_values, bar_width,
               color=colors['unlearn'], edgecolor='black', linewidth=1, label='Unlearn Set')
        ax.bar(x + bar_width/2, retain_values, bar_width,
               color=colors['retain'], edgecolor='black', linewidth=1, label='Retain Set')

        # Plot baselines
        if data['pretrain']['unlearn']:
            ax.axhline(y=data['pretrain']['unlearn'][metric_key], color=colors['Pretrain'],
                       linestyle='--', linewidth=2, alpha=0.8)
        if data['retrain']['unlearn']:
            ax.axhline(y=data['retrain']['unlearn'][metric_key], color=colors['Retrain'],
                       linestyle=':', linewidth=2, alpha=0.8)

        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xlabel('Unlearn Epoch', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(unlearn_epochs)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Adjust y-axis for reward
        if metric_key == 'avg_reward':
            all_vals = [v for v in unlearn_values + retain_values if not np.isnan(v)]
            if all_vals:
                y_min, y_max = min(all_vals), max(all_vals)
                if data['pretrain']['unlearn']:
                    y_min = min(y_min, data['pretrain']['unlearn'][metric_key])
                    y_max = max(y_max, data['pretrain']['unlearn'][metric_key])
                if data['retrain']['unlearn']:
                    y_min = min(y_min, data['retrain']['unlearn'][metric_key])
                    y_max = max(y_max, data['retrain']['unlearn'][metric_key])
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
                ax.set_ylim(y_min - margin, y_max + margin)

    # Legend
    legend_elements = [
        Patch(facecolor=colors['unlearn'], edgecolor='black', label='Unlearn Set'),
        Patch(facecolor=colors['retain'], edgecolor='black', label='Retain Set'),
        Line2D([0], [0], color=colors['Pretrain'], linestyle='--', linewidth=2, label='Pretrain'),
        Line2D([0], [0], color=colors['Retrain'], linestyle=':', linewidth=2, label='Retrain')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 0.98), fontsize=12, framealpha=0.9)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_file}")
    plt.close()


def main():
    """Generate DFRU unlearn vs retain comparison plots for all steps."""
    print("=" * 80)
    print("DFRU: Unlearn Set vs Retain Set Comparison")
    print(f"Processing steps: {STEPS_LIST}")
    print("=" * 80)

    for step in STEPS_LIST:
        print(f"\nProcessing Step = {step}")
        xlsx_file = DFRU_XLSX_TEMPLATE.format(step=step)
        output_file = DFRU_OUTPUT_TEMPLATE.format(step=step)

        data = load_dfru_data(xlsx_file, UNLEARN_EPOCHS)
        if data:
            plot_dfru_comparison(data, UNLEARN_EPOCHS, output_file, step)
        else:
            print(f"  Skipped: No data available")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
