import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
import os
from matplotlib.patches import Rectangle


# Color scheme for map elements
COLORS = {
    'background': '#f0f0f0',
    'obstacle_type_1': '#e67e22',  # 10% penalty
    'obstacle_type_2': '#2c3e50',  # 20% penalty
    'obstacle_type_3': '#8e44ad',  # 30% penalty
    'obstacle_type_4': '#c0392b',  # 40% penalty
    'obstacle_boundary': '#34495e',
    'target': '#e74c3c',
    'start': '#27ae60',
    'grid': '#bdc3c7'
}

# Symbols for different obstacle types
OBSTACLE_SYMBOLS = {
    'type_1': '@',
    'type_2': '#',
    'type_3': '$',
    'type_4': '%'
}


def load_map_data(map_file):
    """Load and convert map data from JSON file.

    Args:
        map_file: Path to the map JSON file

    Returns:
        List of map dictionaries with obstacles and target locations
    """
    with open(map_file, 'r') as f:
        maps_data = json.load(f)

    maps = []
    for map_data in maps_data:
        obstacles_dict = {
            f'type_{i}': [tuple(obs) for obs in map_data[f'obstacles_type_{i}']]
            for i in range(1, 5)
        }
        obstacles_dict['type_boundary'] = [
            tuple(obs) for obs in map_data['obstacles_type_boundary']
        ]

        maps.append({
            "obstacles_location": obstacles_dict,
            "target_location": tuple(map_data["target_location"])
        })

    return maps


def draw_map_grid(ax, map_data, game_type, size=10):
    """Draw a single map on the given axis.

    Args:
        ax: Matplotlib axis object
        map_data: Dictionary containing map information
        game_type: Type of game environment
        size: Map size (default 10x10)
    """
    start_location = (size // 2, size - 1)
    obstacles = map_data["obstacles_location"]
    target = map_data["target_location"]

    # Draw background grid
    for x in range(size):
        for y in range(size):
            rect = Rectangle((x, y), 1, 1,
                           facecolor=COLORS['background'],
                           edgecolor=COLORS['grid'],
                           linewidth=0.5)
            ax.add_patch(rect)

    # Draw obstacles
    for obs_type, obs_list in obstacles.items():
        if obs_type == 'type_boundary':
            continue

        for obs_x, obs_y in obs_list:
            if 0 <= obs_x < size and 0 <= obs_y < size:
                rect = Rectangle((obs_x, obs_y), 1, 1,
                               facecolor=COLORS[f'obstacle_{obs_type}'],
                               edgecolor='black',
                               linewidth=1)
                ax.add_patch(rect)

                symbol = OBSTACLE_SYMBOLS[obs_type]
                ax.text(obs_x + 0.5, obs_y + 0.5, symbol,
                       ha='center', va='center',
                       fontsize=12, fontweight='bold',
                       color='white')

    # Draw target
    target_x, target_y = target
    if 0 <= target_x < size and 0 <= target_y < size:
        rect = Rectangle((target_x, target_y), 1, 1,
                        facecolor=COLORS['target'],
                        edgecolor='darkred',
                        linewidth=2)
        ax.add_patch(rect)
        ax.text(target_x + 0.5, target_y + 0.5, 'T',
               ha='center', va='center',
               fontsize=14, fontweight='bold',
               color='white')

    # Draw start position
    start_x, start_y = start_location
    if 0 <= start_x < size and 0 <= start_y < size:
        rect = Rectangle((start_x, start_y), 1, 1,
                        facecolor=COLORS['start'],
                        edgecolor='darkgreen',
                        linewidth=2)
        ax.add_patch(rect)
        ax.text(start_x + 0.5, start_y + 0.5, 'S',
               ha='center', va='center',
               fontsize=14, fontweight='bold',
               color='white')

    # Configure axis
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)


def visualize_all_maps(map_file, game_type=None, output_file=None,
                       maps_per_row=10, size=None):
    """Visualize all maps in a grid layout.

    Args:
        map_file: Path to map JSON file
        game_type: Type of game ("grid_world" or "aircraft_landing")
        output_file: Output file path (auto-generated if None)
        maps_per_row: Number of maps per row (default 10)
        size: Map grid size (default 10)

    Returns:
        Matplotlib figure object
    """
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(map_file))[0]
        dir_name = os.path.dirname(map_file)
        output_file = os.path.join(dir_name, f"{base_name}.png")

    if not os.path.exists(map_file):
        print(f"Error: Map file not found: {map_file}")
        return

    maps = load_map_data(map_file)
    n_maps = len(maps)
    n_rows = (n_maps + maps_per_row - 1) // maps_per_row

    print(f"Loaded {n_maps} maps")

    # Create figure
    fig_width = maps_per_row * 4
    fig_height = n_rows * 4
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(n_rows, maps_per_row, figure=fig,
                          hspace=0.3, wspace=0.2)

    # Draw each map
    for i, map_data in enumerate(maps):
        row = i // maps_per_row
        col = i % maps_per_row
        ax = fig.add_subplot(gs[row, col])

        draw_map_grid(ax, map_data, game_type, size)
        ax.set_title(f'Map {i}', fontsize=12, fontweight='bold')

    # Hide unused subplots
    for i in range(n_maps, n_rows * maps_per_row):
        row = i // maps_per_row
        col = i % maps_per_row
        ax = fig.add_subplot(gs[row, col])
        ax.set_visible(False)

    # Add title and save
    game_type_name = "Grid World" if game_type == "grid_world" else "Aircraft Landing"
    fig.suptitle(f'{game_type_name} - All Maps Visualization\n'
                 f'Legend: S=Start, T=Target, @=10%, #=20%, $=30%, %=40%',
                 fontsize=16, fontweight='bold', y=0.95)

    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Visualization saved to: {output_file}")

    plt.show()
    return fig


def visualize_maps_for_spu(map_file, game_type, n_maps, size, n_ob):
    """Auto-layout visualization optimized for SPU experiments.

    Args:
        map_file: Path to map file
        game_type: Game environment type
        n_maps: Number of maps (determines layout)

    Returns:
        Matplotlib figure object
    """
    if n_maps <= 20:
        maps_per_row = 4
    elif n_maps <= 50:
        maps_per_row = 10
    else:
        maps_per_row = int(np.ceil(np.sqrt(n_maps)))

    print(f"Using {maps_per_row} columns layout for {n_maps} maps")

    return visualize_all_maps(
        map_file=map_file,
        game_type=game_type,
        maps_per_row=maps_per_row,
        size=size,
    )


def visualize_single_map(map_data, map_index=0, game_type="grid_world",
                         output_file=None, size=10):
    """Visualize a single map.

    Args:
        map_data: Map data dictionary
        map_index: Map index for labeling
        game_type: Game environment type
        output_file: Output file path (optional)
        size: Map grid size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    draw_map_grid(ax, map_data, game_type, size)

    game_type_name = "Grid World" if game_type == "grid_world" else "Aircraft Landing"
    ax.set_title(f'{game_type_name} - Map {map_index}\n'
                f'S=Start, T=Target, @=10%, #=20%, $=30%, %=40%',
                fontsize=14, fontweight='bold')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Single map saved to: {output_file}")

    plt.show()
    return fig


if __name__ == "__main__":
    game_type = "grid_world"
    map_file = f"map_data/{game_type}/50_maps_seed_28.json"

    if os.path.exists(map_file):
        print(f"Map file: {map_file}")
        print(f"Game type: {game_type}")

        fig = visualize_all_maps(
            map_file=map_file,
            game_type=game_type,
            maps_per_row=10
        )
    else:
        print(f"Map file not found: {map_file}")