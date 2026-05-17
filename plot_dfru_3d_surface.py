"""
DFRU 3D Surface Plot Visualization
展示DFRU方法在三维参数空间下的遗忘有效性

三个维度：
- X轴：reward_scale (奖励值) 1-9, 步长2
- Y轴：poison_intensity (投毒比例) 10-80, 步长10
- Z轴：目标障碍物撞击率 (original_unlearn_unlearn_ratio)
- 每个unlearn_epoch生成一个3D曲面图

数据来源：original_unlearn_unlearn_ratio (完整地图的unlearn set部分)
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import argparse
from matplotlib import cm
from matplotlib.colors import Normalize

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
OUTPUT_TEMPLATE = "dfru_3d_surface_step_{step}.png"

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


def create_surface_data(dfru_data, epoch, reward_scales, poison_intensities):
    """
    Create meshgrid data for 3D surface plot.

    Args:
        dfru_data: DataFrame with DFRU results
        epoch: Target unlearn_epoch
        reward_scales: List of reward_scale values
        poison_intensities: List of poison_intensity values

    Returns:
        X, Y, Z: Meshgrid arrays for 3D plotting
    """
    # Create meshgrid
    X, Y = np.meshgrid(reward_scales, poison_intensities)
    Z = np.full_like(X, np.nan, dtype=float)

    epoch_data = dfru_data[dfru_data['unlearn_epoch'] == epoch]

    for _, row in epoch_data.iterrows():
        rs = int(row['reward_scale'])
        pi = int(row['poison_intensity'])

        if rs in reward_scales and pi in poison_intensities:
            rs_idx = reward_scales.index(rs)
            pi_idx = poison_intensities.index(pi)
            Z[pi_idx, rs_idx] = row['original_unlearn_unlearn_ratio']

    return X, Y, Z


def plot_single_3d_surface(data, epoch, output_file, step, vmin=None, vmax=None):
    """
    Create a single 3D surface plot for one epoch.

    Args:
        data: Data dict with pretrain, retrain, and dfru_data
        epoch: Target unlearn_epoch
        output_file: Output filename
        step: Current training step
        vmin, vmax: Color range limits
    """
    if data is None or data['dfru_data'] is None:
        print(f"No DFRU data available for epoch {epoch}")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z = create_surface_data(data['dfru_data'], epoch, REWARD_SCALES, POISON_INTENSITIES)

    # Handle NaN values for plotting
    Z_plot = np.nan_to_num(Z, nan=np.nanmean(Z))

    # Determine color range
    if vmin is None:
        vmin = np.nanmin(Z)
    if vmax is None:
        vmax = np.nanmax(Z)

    # Add baseline values to range
    if data['pretrain']:
        vmax = max(vmax, data['pretrain']['unlearn_ratio'])
    if data['retrain']:
        vmin = min(vmin, data['retrain']['unlearn_ratio'])

    # Create color normalization
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot surface - lower collision rate is better (use reversed colormap)
    surf = ax.plot_surface(X, Y, Z_plot, cmap='RdYlGn_r', norm=norm,
                           edgecolor='black', linewidth=0.3, alpha=0.9,
                           antialiased=True)

    # Add Pretrain baseline plane
    if data['pretrain']:
        pretrain_val = data['pretrain']['unlearn_ratio']
        xx, yy = np.meshgrid(REWARD_SCALES, POISON_INTENSITIES)
        zz_pretrain = np.full_like(xx, pretrain_val, dtype=float)
        ax.plot_surface(xx, yy, zz_pretrain, alpha=0.3, color='blue', label='Pretrain')
        # Add text label
        ax.text(REWARD_SCALES[-1], POISON_INTENSITIES[-1], pretrain_val + 1,
                f'Pretrain: {pretrain_val:.1f}%', color='blue', fontsize=10, fontweight='bold')

    # Add Retrain baseline plane
    if data['retrain']:
        retrain_val = data['retrain']['unlearn_ratio']
        xx, yy = np.meshgrid(REWARD_SCALES, POISON_INTENSITIES)
        zz_retrain = np.full_like(xx, retrain_val, dtype=float)
        ax.plot_surface(xx, yy, zz_retrain, alpha=0.3, color='purple', label='Retrain')
        # Add text label
        ax.text(REWARD_SCALES[0], POISON_INTENSITIES[0], retrain_val - 1,
                f'Retrain: {retrain_val:.1f}%', color='purple', fontsize=10, fontweight='bold')

    # Set labels
    ax.set_xlabel('Reward Scale', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Poison Intensity (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('Target Obstacle Collision Rate (%)', fontsize=12, fontweight='bold', labelpad=10)

    ax.set_title(f'DFRU Unlearn Effectiveness (3D Surface)\n'
                 f'Epoch = {epoch}, Training Steps = {step}',
                 fontsize=14, fontweight='bold', pad=20)

    # Set tick labels
    ax.set_xticks(REWARD_SCALES)
    ax.set_yticks(POISON_INTENSITIES)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('Collision Rate (%)', fontsize=11, fontweight='bold')

    # Set view angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()

    # Save figure
    output_name = output_file.replace('.png', f'_epoch_{epoch}.png')
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_name}")
    plt.close()


def plot_all_epochs_3d_surface(data, output_file, step):
    """
    Create 3D surface subplots for all epochs.

    Args:
        data: Data dict with pretrain, retrain, and dfru_data
        output_file: Output filename
        step: Current training step
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

    # Get global min/max for consistent colorbar
    all_values = []
    for epoch in epochs_to_plot:
        _, _, Z = create_surface_data(data['dfru_data'], epoch, REWARD_SCALES, POISON_INTENSITIES)
        valid_values = Z[~np.isnan(Z)]
        all_values.extend(valid_values)

    if len(all_values) == 0:
        print("No valid data for surface plot")
        return

    vmin = min(all_values)
    vmax = max(all_values)

    if data['pretrain']:
        vmax = max(vmax, data['pretrain']['unlearn_ratio'])
    if data['retrain']:
        vmin = min(vmin, data['retrain']['unlearn_ratio'])

    print(f"Z-axis range: {vmin:.2f}% - {vmax:.2f}%")

    # Determine subplot layout (2 rows x 4 cols for 8 epochs)
    n_epochs = len(epochs_to_plot)
    n_cols = 4
    n_rows = (n_epochs + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 5 * n_rows))
    fig.suptitle(f'DFRU Unlearn Effectiveness: 3D Surface Plots\n',
                 fontsize=16, fontweight='bold', y=1.02)

    # Create color normalization
    norm = Normalize(vmin=vmin, vmax=vmax)

    for idx, epoch in enumerate(epochs_to_plot):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        X, Y, Z = create_surface_data(data['dfru_data'], epoch, REWARD_SCALES, POISON_INTENSITIES)
        Z_plot = np.nan_to_num(Z, nan=np.nanmean(Z))

        # Plot surface
        surf = ax.plot_surface(X, Y, Z_plot, cmap='RdYlGn_r', norm=norm,
                               edgecolor='black', linewidth=0.2, alpha=0.9,
                               antialiased=True)

        # Add Pretrain baseline plane (semi-transparent)
        if data['pretrain']:
            pretrain_val = data['pretrain']['unlearn_ratio']
            xx, yy = np.meshgrid(REWARD_SCALES, POISON_INTENSITIES)
            zz_pretrain = np.full_like(xx, pretrain_val, dtype=float)
            ax.plot_surface(xx, yy, zz_pretrain, alpha=0.2, color='blue')

        # Add Retrain baseline plane (semi-transparent)
        if data['retrain']:
            retrain_val = data['retrain']['unlearn_ratio']
            xx, yy = np.meshgrid(REWARD_SCALES, POISON_INTENSITIES)
            zz_retrain = np.full_like(xx, retrain_val, dtype=float)
            ax.plot_surface(xx, yy, zz_retrain, alpha=0.2, color='purple')

        # Set labels
        ax.set_xlabel('Reward Scale', fontsize=9, labelpad=5)
        ax.set_ylabel('Poison Intensity', fontsize=9, labelpad=5)
        ax.set_zlabel('Collision Rate (%)', fontsize=9, labelpad=5)
        ax.set_title(f'Epoch = {epoch}', fontsize=12, fontweight='bold')

        # Set consistent Z-axis range
        ax.set_zlim(vmin - 2, vmax + 2)

        # Set view angle
        ax.view_init(elev=25, azim=45)

        # Smaller tick labels for subplots
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Target Obstacle Collision Rate (%)', fontsize=12, fontweight='bold')

    # Add baseline legend with color blocks
    from matplotlib.patches import Rectangle

    legend_x = 0.02
    legend_y = 0.02
    block_width = 0.025
    block_height = 0.018
    text_offset = 0.03
    spacing = 0.14

    legend_items = []
    if data['pretrain']:
        legend_items.append(('Pretrain', data['pretrain']['unlearn_ratio'], 'blue'))
    if data['retrain']:
        legend_items.append(('Retrain', data['retrain']['unlearn_ratio'], 'purple'))

    for i, (label, value, color) in enumerate(legend_items):
        x_pos = legend_x + i * spacing
        # Draw color block
        rect = Rectangle((x_pos, legend_y), block_width, block_height,
                         transform=fig.transFigure, facecolor=color,
                         edgecolor='black', linewidth=1, clip_on=False, alpha=0.7)
        fig.add_artist(rect)
        # Add text label
        fig.text(x_pos + text_offset, legend_y + block_height / 2,
                f'{label}: {value:.2f}%', fontsize=11, fontweight='bold',
                va='center', transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_file}")
    plt.close()


def plot_animated_view(data, epoch, output_prefix, step):
    """
    Create multiple views of the 3D surface for a single epoch.
    Generates images from different angles.

    Args:
        data: Data dict with pretrain, retrain, and dfru_data
        epoch: Target unlearn_epoch
        output_prefix: Output filename prefix
        step: Current training step
    """
    if data is None or data['dfru_data'] is None:
        print(f"No DFRU data available for epoch {epoch}")
        return

    # Different viewing angles
    angles = [(25, 45), (25, 135), (25, 225), (25, 315), (45, 45), (60, 45)]

    for elev, azim in angles:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        X, Y, Z = create_surface_data(data['dfru_data'], epoch, REWARD_SCALES, POISON_INTENSITIES)
        Z_plot = np.nan_to_num(Z, nan=np.nanmean(Z))

        vmin = np.nanmin(Z)
        vmax = np.nanmax(Z)
        if data['pretrain']:
            vmax = max(vmax, data['pretrain']['unlearn_ratio'])
        if data['retrain']:
            vmin = min(vmin, data['retrain']['unlearn_ratio'])

        norm = Normalize(vmin=vmin, vmax=vmax)

        surf = ax.plot_surface(X, Y, Z_plot, cmap='RdYlGn_r', norm=norm,
                               edgecolor='black', linewidth=0.3, alpha=0.9)

        if data['pretrain']:
            xx, yy = np.meshgrid(REWARD_SCALES, POISON_INTENSITIES)
            zz = np.full_like(xx, data['pretrain']['unlearn_ratio'], dtype=float)
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue')

        if data['retrain']:
            xx, yy = np.meshgrid(REWARD_SCALES, POISON_INTENSITIES)
            zz = np.full_like(xx, data['retrain']['unlearn_ratio'], dtype=float)
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='purple')

        ax.set_xlabel('Reward Scale', fontsize=11, fontweight='bold')
        ax.set_ylabel('Poison Intensity (%)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Collision Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'DFRU 3D Surface (Epoch={epoch}, View: elev={elev}, azim={azim})',
                    fontsize=12, fontweight='bold')

        ax.view_init(elev=elev, azim=azim)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        plt.tight_layout()

        output_name = f"{output_prefix}_epoch_{epoch}_view_{elev}_{azim}.png"
        plt.savefig(output_name, dpi=200, bbox_inches='tight')
        print(f"Figure saved: {output_name}")
        plt.close()


def main():
    """Main function to generate DFRU 3D surface visualizations."""
    parser = argparse.ArgumentParser(description='DFRU 3D Surface Plot Visualization')
    parser.add_argument('--tem_dfru_max_steps', type=int, default=25,
                        help='Training steps (default: 25)')
    parser.add_argument('--single_epoch', type=int, default=None,
                        help='Generate single epoch 3D surface (optional)')
    parser.add_argument('--multi_view', action='store_true',
                        help='Generate multiple viewing angles for single epoch')
    args = parser.parse_args()

    step = args.tem_dfru_max_steps

    print("=" * 80)
    print("DFRU 3D Surface Plot Visualization")
    print(f"Training Steps: {step}")
    print("=" * 80)

    # Load data
    xlsx_file = DFRU_XLSX_TEMPLATE.format(step=step)
    print(f"\nLoading data from: {xlsx_file}")
    data = load_dfru_data(xlsx_file)

    if data is None:
        print("Failed to load data. Exiting.")
        return

    output_file = OUTPUT_TEMPLATE.format(step=step)

    # Generate plots
    if args.single_epoch is not None:
        if args.multi_view:
            # Multiple viewing angles
            output_prefix = output_file.replace('.png', '')
            plot_animated_view(data, args.single_epoch, output_prefix, step)
        else:
            # Single 3D surface
            plot_single_3d_surface(data, args.single_epoch, output_file, step)
    else:
        # All epochs overview
        plot_all_epochs_3d_surface(data, output_file, step)

        # Also generate individual epoch plots
        print("\nGenerating individual epoch 3D surfaces...")
        for epoch in UNLEARN_EPOCHS:
            if data['dfru_data'] is not None and epoch in data['dfru_data']['unlearn_epoch'].values:
                plot_single_3d_surface(data, epoch, output_file, step)

    print("\n" + "=" * 80)
    print("Visualization completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
