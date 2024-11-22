#create network graph

import pygraphviz as pgv
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import os 

from PIL import Image
from io import BytesIO

# Add the parent directory of 'network' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from network.model import Network

os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_path = r"../data/test_network_week_45_bigger.csv"
df_network = pd.read_csv(file_path, sep=',', encoding='utf-8')


#initialize the network
network = Network()
network.initate_network(df_network)

# Fetch delay matrix at a specific time
current_delay_matrix = network.fetch_D_matrix(df_network)

# Define color map for delays
color_map = {
    # Blue shades for negative values
    -15: "#0D47A1",  # Dark Blue (Very Large Negative)
    -10: "#1976D2",  # Deep Blue
    -5: "#42A5F5",   # Medium Blue
    -3: "#90CAF9",   # Light Blue
    -1: "#E3F2FD",   # Very Light Blue (Close to 0)

    # Red shades for positive values
    1: "#FFEBEE",    # Very Light Red (Close to 0)
    3: "#FFCDD2",    # Light Red
    5: "#E57373",    # Medium Red
    10: "#D32F2F",   # Deep Red
    15: "#B71C1C"    # Dark Red (Very Large Positive)
}

# Function to get color based on value
def get_color(value):
    if value == 0:
        return "#D3D3D3"

    if value < 0:
        for key in sorted(color_map.keys(), reverse=True):
            if value >= key:
                return color_map[key]
    else:
        for key in sorted(color_map.keys()):
            if value <= key:
                return color_map[key]
    return color_map[key]

# Number of time steps to visualize
time_steps = 10

    
# Visualization for time steps
for step in range(time_steps):
    print(f"Visualizing delay for time step {step}")
    print("Network time: ", network.current_time)
    network.print_delay_matrix(print_all=False)
    
    if step != 0:
        network.predict_time_step()

    # Get the updated delay matrix
    delay_matrix = network.D_matrix
    
    # Aggregate delays into a dictionary for visualization
    station_delays = {station: delay_matrix[idx][0] for station, idx in network.station_indicies.items()}
    

    # Create the PyGraphviz graph
    G = pgv.AGraph(directed=True)

    # Add nodes with colors based on delays
    for station, delay in station_delays.items():
        color = get_color(delay)
        G.add_node(station, style="filled", fillcolor=color, fontsize=10,margin="0.2,0.2")

    # Add edges
    for (start, end), edge in network.edges.items():
        G.add_edge(start, end, penwidth=2, color="gray")  

    # Adjust layout parameters for spacing
    G.graph_attr.update(
        rankdir="LR",  # Layout direction: Left-to-Right; can also use "TB" (Top-to-Bottom)
        nodesep="2.0",  # Minimum space between nodes
        ranksep="1.5",  # Minimum vertical space between ranks (layers of nodes)
        splines="true", # Use curved edges for better aesthetics
        dpi="300"       # Increase resolution for clarity

    )   

    G.layout(prog="neato")  # Use the 'neato' layout for better node spacing

    # Save the graph to a file for viewing
    output_path = f"network_delay_step_{step + 1}.png"
    G.draw(output_path, format="png")
    print(f"Graph saved to {output_path}")



