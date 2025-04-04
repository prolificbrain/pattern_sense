"""Generate a logo for PatternSense."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle, Polygon
import os

# Set up the figure
plt.figure(figsize=(10, 10), dpi=300)
ax = plt.subplot(111)
ax.set_aspect('equal')

# Create a custom colormap for the background gradient
colors = [(0.05, 0.1, 0.2), (0.1, 0.2, 0.3), (0.05, 0.1, 0.2)]
cmap = LinearSegmentedColormap.from_list('patternsense_cmap', colors, N=100)

# Create background gradient
x = np.linspace(-1, 1, 1000)
y = np.linspace(-1, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)
plt.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')

# Draw a pattern grid in the background
grid_size = 20
grid_alpha = 0.1
for i in range(-grid_size, grid_size+1, 4):
    plt.plot([i/grid_size, i/grid_size], [-1, 1], color='white', alpha=grid_alpha, linewidth=0.5)
    plt.plot([-1, 1], [i/grid_size, i/grid_size], color='white', alpha=grid_alpha, linewidth=0.5)

# Draw the main hexagon (representing pattern recognition)
hex_radius = 0.7
hex_points = []
for i in range(6):
    angle = i * 2 * np.pi / 6
    x = hex_radius * np.cos(angle)
    y = hex_radius * np.sin(angle)
    hex_points.append((x, y))

hexagon = Polygon(hex_points, fill=False, edgecolor='#3498db', linewidth=4, alpha=0.8)
ax.add_patch(hexagon)

# Draw inner pattern nodes
for i in range(12):
    angle = i * 2 * np.pi / 12
    r = 0.45
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    size = 0.06 + 0.02 * np.sin(i * np.pi / 6)  # Varying sizes
    color = plt.cm.viridis(i / 12)  # Color from viridis colormap
    circle = Circle((x, y), size, fill=True, color=color, alpha=0.8)
    ax.add_patch(circle)

# Draw connecting lines (representing pattern relationships)
for i in range(12):
    angle1 = i * 2 * np.pi / 12
    r1 = 0.45
    x1 = r1 * np.cos(angle1)
    y1 = r1 * np.sin(angle1)
    
    # Connect to 3 neighboring nodes
    for j in range(1, 4):
        idx = (i + j) % 12
        angle2 = idx * 2 * np.pi / 12
        r2 = 0.45
        x2 = r2 * np.cos(angle2)
        y2 = r2 * np.sin(angle2)
        
        plt.plot([x1, x2], [y1, y2], color='white', alpha=0.4, linewidth=1)

# Draw the central node (representing the core)
central_circle = Circle((0, 0), 0.15, fill=True, color='#e74c3c', alpha=0.9)
ax.add_patch(central_circle)

# Add pattern recognition symbols in the center
plt.text(-0.05, 0.02, "P", color='white', fontsize=14, fontweight='bold')
plt.text(0.02, 0.02, "S", color='white', fontsize=14, fontweight='bold')

# Add title
plt.text(0, -0.95, "PatternSense", color='white', fontsize=36, fontweight='bold', 
         ha='center', va='center')

# Remove axes
plt.axis('off')

# Save the logo
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patternsense_logo.png')
plt.savefig(output_path, bbox_inches='tight', transparent=True)
plt.close()

print(f"Logo saved to {output_path}")
