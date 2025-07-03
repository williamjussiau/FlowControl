import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_velocity_fields(U_data, V_coords, snapshot_idx=0, grid_resolution=200, figsize=(15, 5)):
    """
    Plot velocity fields from FEniCS simulation data
    
    Parameters:
    -----------
    U_data : np.ndarray
        Velocity data array, shape (n_dofs, n_snapshots, n_files)
    V_coords : np.ndarray
        Velocity DOF coordinates, shape (n_dofs, 2)
    snapshot_idx : int
        Which snapshot to plot (default: 0)
    grid_resolution : int
        Grid resolution for interpolation (default: 200)
    figsize : tuple
        Figure size (default: (15, 5))
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Extract data for specific snapshot
    if U_data.ndim == 3:
        U_snapshot = U_data[:, snapshot_idx, 0]
    elif U_data.ndim == 2:
        U_snapshot = U_data[:, snapshot_idx]
    else:
        U_snapshot = U_data
    
    # Extract u and v values (interleaved DOF format)
    u_vals = U_snapshot[0::2]  # Every other starting from 0: u1, u2, u3, ...
    v_vals = U_snapshot[1::2]  # Every other starting from 1: v1, v2, v3, ...
    
    # Extract coordinates (also interleaved - take every other)
    coords = V_coords[0::2, :]  # Every other coordinate: coord1, coord2, coord3, ...
    
    # Calculate velocity magnitude
    vel_mag = np.sqrt(u_vals**2 + v_vals**2)
    
    # Create regular grid for imshow
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    # Create grid
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate data onto regular grid
    vel_mag_grid = griddata(coords, vel_mag, (Xi, Yi), method='linear')
    u_vals_grid = griddata(coords, u_vals, (Xi, Yi), method='linear')
    v_vals_grid = griddata(coords, v_vals, (Xi, Yi), method='linear')
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot velocity magnitude
    im1 = axes[0].imshow(vel_mag_grid, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                         cmap='viridis', aspect='equal')
    axes[0].set_title('Velocity Magnitude')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot u component
    im2 = axes[1].imshow(u_vals_grid, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                         cmap='RdBu_r', aspect='equal')
    axes[1].set_title('u-velocity')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot v component
    im3 = axes[2].imshow(v_vals_grid, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                         cmap='RdBu_r', aspect='equal')
    axes[2].set_title('v-velocity')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    print(f"Grid resolution: {grid_resolution}x{grid_resolution}")
    print(f"Domain: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
    print(f"Snapshot: {snapshot_idx}")
    
    return fig, axes

def plot_velocity_magnitude_only(U_data, V_coords, snapshot_idx=0, grid_resolution=200, figsize=(8, 6)):
    """
    Plot only velocity magnitude for a cleaner single plot
    """
    
    # Extract data for specific snapshot
    if U_data.ndim == 3:
        U_snapshot = U_data[:, snapshot_idx, 0]
    elif U_data.ndim == 2:
        U_snapshot = U_data[:, snapshot_idx]
    else:
        U_snapshot = U_data
    
    # Extract u and v values (interleaved DOF format)
    u_vals = U_snapshot[0::2]
    v_vals = U_snapshot[1::2]
    coords = V_coords[0::2, :]
    
    # Calculate velocity magnitude
    vel_mag = np.sqrt(u_vals**2 + v_vals**2)
    
    # Create regular grid
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate data onto regular grid
    vel_mag_grid = griddata(coords, vel_mag, (Xi, Yi), method='linear')
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(vel_mag_grid, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                   cmap='viridis', aspect='equal')
    ax.set_title(f'Velocity Magnitude (Snapshot {snapshot_idx})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='Velocity Magnitude')
    
    plt.tight_layout()
    
    return fig, ax