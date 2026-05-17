"""
DFRU 3D Heatmap Visualization
展示DFRU方法在三维参数空间下的遗忘有效性

三个维度：
- X轴：reward_scale (奖励值) 1-9, 步长2
- Y轴：poison_intensity (投毒比例) 10-80, 步长10
- 颜色深浅：目标障碍物撞击率 (original_unlearn_unlearn_ratio)
- 每个unlearn_epoch生成一个子图

数据来源：original_unlearn_unlearn_ratio (完整地图的unlearn set部分)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Plot configuration
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

################################################################################
# Configuration
################################################################################

# 参数网格
UNLEARN_EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40]
REWARD_SCALES = [1, 3, 5, 7, 9]
POISON_INTENSITIES = [10, 20, 30, 40, 50, 60, 70, 80]

# 文件模板
DFRU_XLSX_TEMPLATE = "./result/grid_world/DFRU_Model_standard_epoch_50/tem_dfru_max_steps_{step}/dfru_verification_filtered.xlsx"
OUTPUT_TEMPLATE = "dfru_heatmap_unlearn_ratio_step_{step}.png"

################################################################################


def load_dfru_data(xlsx_file):
    """
    Load DFRU verification results from xlsx file.

    Returns:
        dict: {
            'pretrain': {'unlearn_ratio': float},
            'retrain': {'unlearn_ratio': float},
            'dfru_data': DataFrame with DFRU results
        }
    """
    if not os.path.exists(xlsx_file):
        print(f"Warning: File not found: {xlsx_file}")
        return None

    try:
        df = pd.read_excel(xlsx_file)
        print(f"Loaded DFRU data: {len(df)} rows from {xlsx_file}")
    except Exception as e:
        print(f"Error reading {xlsx_file}: {e}")
        return None

    result = {
        'pretrain': None,
        'retrain': None,
        'dfru_data': None
    }

    # Extract Pretrain baseline
    pretrain_rows = df[df['model_type'] == 'Pretrain']
    if len(pretrain_rows) > 0:
        row = pretrain_rows.iloc[0]
        result['pretrain'] = {
            'unlearn_ratio': float(row['original_unlearn_unlearn_ratio'])
        }
        print(f"  Pretrain unlearn_ratio: {result['pretrain']['unlearn_ratio']:.2f}%")

    # Extract Retrain baseline
    retrain_rows = df[df['model_type'] == 'Retrain']
    if len(retrain_rows) > 0:
        row = retrain_rows.iloc[0]
        result['retrain'] = {
            'unlearn_ratio': float(row['original_unlearn_unlearn_ratio'])
        }
        print(f"  Retrain unlearn_ratio: {result['retrain']['unlearn_ratio']:.2f}%")

    # Extract DFRU data
    dfru_rows = df[df['model_type'] == 'DFRU']
    if len(dfru_rows) > 0:
        result['dfru_data'] = dfru_rows[['unlearn_epoch', 'reward_scale', 'poison_intensity',
                                          'original_unlearn_unlearn_ratio', 'status']].copy()
        # Filter only successful models
        result['dfru_data'] = result['dfru_data'][result['dfru_data']['status'] == 'success']
        print(f"  DFRU models loaded: {len(result['dfru_data'])} successful")

    return result


def create_heatmap_matrix(dfru_data, epoch, reward_scales, poison_intensities):
    """
    Create 2D matrix for heatmap from DFRU data for a specific epoch.

    Args:
        dfru_data: DataFrame with DFRU results
        epoch: Target unlearn_epoch
        reward_scales: List of reward_scale values
        poison_intensities: List of poison_intensity values

    Returns:
        2D numpy array (poison_intensity x reward_scale)
    """
    matrix = np.full((len(poison_intensities), len(reward_scales)), np.nan)

    epoch_data = dfru_data[dfru_data['unlearn_epoch'] == epoch]

    for _, row in epoch_data.iterrows():
        rs = int(row['reward_scale'])
        pi = int(row['poison_intensity'])

        if rs in reward_scales and pi in poison_intensities:
            rs_idx = reward_scales.index(rs)
            pi_idx = poison_intensities.index(pi)
            matrix[pi_idx, rs_idx] = row['original_unlearn_unlearn_ratio']

    return matrix


def plot_dfru_heatmaps(data, output_file, step):
    """
    Create heatmap subplots for each unlearn_epoch.

    Args:
        data: Data dict with pretrain, retrain, and dfru_data
        output_file: Output filename
        step: Current training step (for title)
    """
    if data is None or data['dfru_data'] is None or len(data['dfru_data']) == 0:
        print("No DFRU data available for plotting")
        return

    # Filter epochs that have data
    available_epochs = sorted(data['dfru_data']['unlearn_epoch'].unique())
    epochs_to_plot = [e for e in UNLEARN_EPOCHS if e in available_epochs]

    if len(epochs_to_plot) == 0:
        print("No epochs with data found")
        return

    print(f"Plotting epochs: {epochs_to_plot}")

    # Determine subplot layout (2 rows x 4 cols for 8 epochs)
    n_epochs = len(epochs_to_plot)
    n_cols = 4
    n_rows = (n_epochs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(f'DFRU Unlearn Effectiveness: Target Obstacle Collision Rate (%) on Unlearn Set\n'
                 f'Training Steps = {step}',
                 fontsize=14, fontweight='bold', y=1.02)

    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Get global min/max for consistent colorbar
    all_values = []
    for epoch in epochs_to_plot:
        matrix = create_heatmap_matrix(data['dfru_data'], epoch, REWARD_SCALES, POISON_INTENSITIES)
        valid_values = matrix[~np.isnan(matrix)]
        all_values.extend(valid_values)

    if len(all_values) == 0:
        print("No valid data for heatmap")
        return

    vmin = min(all_values)
    vmax = max(all_values)

    # Add baseline values to range for context
    if data['pretrain']:
        vmax = max(vmax, data['pretrain']['unlearn_ratio'])
    if data['retrain']:
        vmin = min(vmin, data['retrain']['unlearn_ratio'])

    print(f"Colorbar range: {vmin:.2f}% - {vmax:.2f}%")

    # Plot each epoch
    im = None
    for idx, epoch in enumerate(epochs_to_plot):
        ax = axes_flat[idx]

        matrix = create_heatmap_matrix(data['dfru_data'], epoch, REWARD_SCALES, POISON_INTENSITIES)

        # Create heatmap - lower collision rate is better (use reversed colormap)
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                       vmin=vmin, vmax=vmax, origin='lower')

        # Set axis labels
        ax.set_xticks(range(len(REWARD_SCALES)))
        ax.set_xticklabels(REWARD_SCALES)
        ax.set_yticks(range(len(POISON_INTENSITIES)))
        ax.set_yticklabels(POISON_INTENSITIES)

        ax.set_xlabel('Reward Scale', fontsize=10)
        ax.set_ylabel('Poison Intensity (%)', fontsize=10)
        ax.set_title(f'Epoch = {epoch}', fontsize=12, fontweight='bold')

        # Add value annotations
        for i in range(len(POISON_INTENSITIES)):
            for j in range(len(REWARD_SCALES)):
                value = matrix[i, j]
                if not np.isnan(value):
                    # Choose text color based on background
                    text_color = 'white' if value > (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                           fontsize=8, color=text_color, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(epochs_to_plot), len(axes_flat)):
        axes_flat[idx].axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Target Obstacle Collision Rate (%)', fontsize=11, fontweight='bold')

    # Add baseline annotations with color blocks
    from matplotlib.patches import Rectangle
    import matplotlib.cm as cm

    # Get colormap for consistent colors
    cmap = cm.get_cmap('RdYlGn_r')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    legend_x = 0.02
    legend_y = 0.02
    block_width = 0.025
    block_height = 0.018
    text_offset = 0.03
    spacing = 0.08

    legend_items = []
    if data['pretrain']:
        color = cmap(norm(data['pretrain']['unlearn_ratio']))
        legend_items.append(('Pretrain', data['pretrain']['unlearn_ratio'], color))
    if data['retrain']:
        color = cmap(norm(data['retrain']['unlearn_ratio']))
        legend_items.append(('Retrain', data['retrain']['unlearn_ratio'], color))

    for i, (label, value, color) in enumerate(legend_items):
        x_pos = legend_x + i * spacing
        # Draw color block
        rect = Rectangle((x_pos, legend_y), block_width, block_height,
                         transform=fig.transFigure, facecolor=color,
                         edgecolor='black', linewidth=1, clip_on=False)
        fig.add_artist(rect)
        # Add text label
        fig.text(x_pos + text_offset, legend_y + block_height / 2,
                f'{label}: {value:.2f}%', fontsize=10, fontweight='bold',
                va='center', transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_file}")
    plt.close()


def plot_single_epoch_heatmap(data, epoch, output_file, step):
    """
    Create a single large heatmap for one epoch.

    Args:
        data: Data dict with pretrain, retrain, and dfru_data
        epoch: Target unlearn_epoch
        output_file: Output filename
        step: Current training step
    """
    if data is None or data['dfru_data'] is None:
        print(f"No DFRU data available for epoch {epoch}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    matrix = create_heatmap_matrix(data['dfru_data'], epoch, REWARD_SCALES, POISON_INTENSITIES)

    # Get value range
    valid_values = matrix[~np.isnan(matrix)]
    if len(valid_values) == 0:
        print(f"No valid data for epoch {epoch}")
        return

    vmin = min(valid_values)
    vmax = max(valid_values)

    if data['pretrain']:
        vmax = max(vmax, data['pretrain']['unlearn_ratio'])
    if data['retrain']:
        vmin = min(vmin, data['retrain']['unlearn_ratio'])

    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=vmin, vmax=vmax, origin='lower')

    # Set axis labels
    ax.set_xticks(range(len(REWARD_SCALES)))
    ax.set_xticklabels(REWARD_SCALES, fontsize=12)
    ax.set_yticks(range(len(POISON_INTENSITIES)))
    ax.set_yticklabels(POISON_INTENSITIES, fontsize=12)

    ax.set_xlabel('Reward Scale', fontsize=14, fontweight='bold')
    ax.set_ylabel('Poison Intensity (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'DFRU Target Obstacle Collision Rate (%) on Unlearn Set\n'
                 f'Epoch = {epoch}, Training Steps = {step}',
                 fontsize=14, fontweight='bold')

    # Add value annotations
    for i in range(len(POISON_INTENSITIES)):
        for j in range(len(REWARD_SCALES)):
            value = matrix[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > (vmin + vmax) / 2 else 'black'
                ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                       fontsize=11, color=text_color, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Target Obstacle Collision Rate (%)', fontsize=12, fontweight='bold')

    # Add baseline lines on colorbar
    if data['pretrain']:
        cbar.ax.axhline(y=(data['pretrain']['unlearn_ratio'] - vmin) / (vmax - vmin),
                        color='blue', linestyle='--', linewidth=2, label='Pretrain')
    if data['retrain']:
        cbar.ax.axhline(y=(data['retrain']['unlearn_ratio'] - vmin) / (vmax - vmin),
                        color='purple', linestyle=':', linewidth=2, label='Retrain')

    # Add baseline annotations with color blocks
    from matplotlib.patches import Rectangle
    import matplotlib.cm as cm

    cmap = cm.get_cmap('RdYlGn_r')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    legend_x = 0.02
    legend_y = -0.12
    block_width = 0.04
    block_height = 0.035
    text_offset = 0.05
    spacing = 0.22

    legend_items = []
    if data['pretrain']:
        color = cmap(norm(data['pretrain']['unlearn_ratio']))
        legend_items.append(('Pretrain', data['pretrain']['unlearn_ratio'], color))
    if data['retrain']:
        color = cmap(norm(data['retrain']['unlearn_ratio']))
        legend_items.append(('Retrain', data['retrain']['unlearn_ratio'], color))

    for i, (label, value, color) in enumerate(legend_items):
        x_pos = legend_x + i * spacing
        # Draw color block
        rect = Rectangle((x_pos, legend_y), block_width, block_height,
                         transform=ax.transAxes, facecolor=color,
                         edgecolor='black', linewidth=1, clip_on=False)
        ax.add_patch(rect)
        # Add text label
        ax.text(x_pos + text_offset, legend_y + block_height / 2,
                f'{label}: {value:.2f}%', fontsize=11, fontweight='bold',
                va='center', transform=ax.transAxes)

    plt.tight_layout()

    # Save figure
    output_name = output_file.replace('.png', f'_epoch_{epoch}.png')
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_name}")
    plt.close()


def main():
    """Main function to generate DFRU heatmap visualizations."""
    parser = argparse.ArgumentParser(description='DFRU Heatmap Visualization')
    parser.add_argument('--tem_dfru_max_steps', type=int, default=25,
                        help='Training steps (default: 25)')
    parser.add_argument('--single_epoch', type=int, default=None,
                        help='Generate single epoch heatmap (optional)')
    args = parser.parse_args()

    step = args.tem_dfru_max_steps

    print("=" * 80)
    print("DFRU Heatmap Visualization")
    print(f"Training Steps: {step}")
    print("=" * 80)

    # Load data
    xlsx_file = DFRU_XLSX_TEMPLATE.format(step=step)
    print(f"\nLoading data from: {xlsx_file}")
    data = load_dfru_data(xlsx_file)

    if data is None:
        print("Failed to load data. Exiting.")
        return

    # Generate plots
    if args.single_epoch is not None:
        # Single epoch heatmap
        output_file = OUTPUT_TEMPLATE.format(step=step)
        plot_single_epoch_heatmap(data, args.single_epoch, output_file, step)
    else:
        # All epochs overview
        output_file = OUTPUT_TEMPLATE.format(step=step)
        plot_dfru_heatmaps(data, output_file, step)

        # Also generate individual epoch plots
        print("\nGenerating individual epoch heatmaps...")
        for epoch in UNLEARN_EPOCHS:
            if data['dfru_data'] is not None and epoch in data['dfru_data']['unlearn_epoch'].values:
                output_file = OUTPUT_TEMPLATE.format(step=step)
                plot_single_epoch_heatmap(data, epoch, output_file, step)

    print("\n" + "=" * 80)
    print("Visualization completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
