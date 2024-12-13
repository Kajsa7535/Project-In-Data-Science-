import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygraphviz as pgv
import copy


# function to create a dictionary with 24 intervals, one for every hour, the value of each interval is 0 and the key is a tuple with the start and end hour of the interval 
def create_24_hour_dict(matrix_element= False, n = None):
    dict = {}

    for i in range(24): # creates 24 intervals, one for every hour
        start_time = pd.to_datetime(f"2019-03-31 {i}:00:00.000").time()
        if i == 23:
            end_time = pd.to_datetime(f"2019-03-31 00:00:00.000").time()
        else:
            end_time = pd.to_datetime(f"2019-03-31 {i+1}:00:00.000").time()
        
        if matrix_element and n:
            dict[(start_time, end_time)] = np.zeros((n, n), dtype=float)
        else:
            dict[(start_time, end_time)] = 0 
    return dict


### Functions specific to create the directed adjacency matrix start
  
#Function that removes the edge in the opposite direction of the delay.
def remove_initial_edge(directed_A_matrix, directed_edge, removed_edges):
    reversed_edge = (directed_edge[1], directed_edge[0])
    directed_A_matrix.loc[reversed_edge[0], reversed_edge[1]] = 0
    removed_edges.append(reversed_edge)
    return directed_A_matrix

#Function that process the current edge: remove incoming edges and update outgoing edges.
def process_current_edge(directed_A_matrix, current_edge, removed_edges, removed_edges_with_count):
    start_node, end_node = current_edge
    # Save the original value of the edge being processed
    original_value = directed_A_matrix.loc[start_node, end_node]
    # Identify incoming edges to the end node
    incoming_edges = directed_A_matrix.loc[:, end_node]
    incoming_nodes = incoming_edges[incoming_edges > 0].index.tolist()
    incoming_nodes = [node for node in incoming_nodes if node != start_node]
    # Update removed edges and count dictionary
    edges_to_remove = [edge for edge in removed_edges if edge[0] == end_node]
    for edge in edges_to_remove:
        removed_edges.remove(edge)
        removed_edges_with_count[edge] = len(incoming_nodes)
    # Remove all incoming edges except the one from the start node
    directed_A_matrix.loc[:, end_node] = 0
    directed_A_matrix.loc[start_node, end_node] = original_value
    # Add removed incoming edges to the removed edges list
    for node in incoming_nodes:
        removed_edges.append((node, end_node))
    # Update outgoing edges
    outgoing_edges = directed_A_matrix.loc[end_node]
    outgoing_nodes = outgoing_edges[outgoing_edges >= 1].index.tolist()
    outgoing_edge_count = len(outgoing_nodes)
    weight = 1 / outgoing_edge_count if outgoing_edge_count > 0 else 0
    for node in outgoing_nodes:
        directed_A_matrix.loc[end_node, node] += weight
    return directed_A_matrix


# Generate new edges to be added to the frontier.
def get_new_edges_to_frontier(directed_A_matrix, current_edge, visited_nodes, frontier):
    _, end_node = current_edge
    outgoing_edges = directed_A_matrix.loc[end_node]
    outgoing_nodes = outgoing_edges[outgoing_edges >= 1].index.tolist()
    # Avoid revisiting nodes or adding already-queued nodes
    new_edges = [(end_node, node) for node in outgoing_nodes if node not in visited_nodes]
    frontier_nodes = [edge[1] for edge in frontier]
    new_edges = [edge for edge in new_edges if edge[1] not in frontier_nodes]
    return new_edges

### Functions specific to create the directed adjacency matrix end

### Functions specific to calculate G matrices

# Function to calculate the Gij matrix cell value for a given time span
def calculate_G_matrix_values(time_span, A_matrix, removed_edges_keys, row_index, i_station, col_index, station_indices, stations, edges):
        Aji = A_matrix.iloc[col_index,row_index] # Aji value
        pji = 0 # probability of the edge from j to i, initialized to 0
        value_to_add = 0 # value to add to the G matrix value if there are removed edges that should be added to the calculation, initialized to 0 
        
        # getting j station name and object, 
        j_station_name = [key for key, value in station_indices.items() if value == col_index][0]
        j_station = stations[j_station_name]
        Bj = j_station.fetch_Bi(time_span)

        #if there is an edge from j to i, we extraact the values of the edge
        if Aji >= 1: 
            edge_ji = edges[(j_station.name, i_station.name)]
            pji = edge_ji.pij
                  
            Aji = 1 # set the value to 1 to make sure the node is weighted as 1
            
            # extract the value to add from the removed edges, if there are any
            value_to_add = directed_delays_value_to_add(removed_edges_keys, edges, j_station_name, Bj)
                
        #kroneker value should only be 1 if the row and col stations are the same
        kronecker = 1 if col_index == row_index else 0
        #calculating the value of the G matrix at the current row and column
        value = Aji * pji * Bj - Bj * kronecker + value_to_add
        return value

# Calculates the value that will be added to GG value, which comes from an edge that has been removed. This only happens when the model used the flag for directed delays
def directed_delays_value_to_add(removed_edges_keys, edges, j_station_name, Bj):
    value_to_add = 0
    # Goes through the removed edges that has the start station as j_station and should add the new value
    edges_to_add = [x for x in removed_edges_keys if x[0] == j_station_name]
    
    #loop through the removed edges that should add value to the G matrix cell
    for edge in edges_to_add:
        #edges_keys = edges.keys()
        #if edge in edges_keys: #NOTE: Removed but if anything goes wrong, try to put it back again
        edge_object = edges[edge] 
        value_to_add += edge_object.Aij * edge_object.pij * Bj
    return value_to_add

### Functions specific to calculate G matrices end

## Other functions for the model 

def find_current_time_G_matrix( matrix_keys, current_time, G_matrices):
        for i in range(len(matrix_keys)): # Goes through all the time intervalls
            if current_time >= matrix_keys[i][0] and current_time < matrix_keys[i][1]: # checks which interval current time is in 
                G_matrix = G_matrices[matrix_keys[i]]#  Extracts the correct G matrix that is in to the correct interval
                break
        return G_matrix


# function to get a color based on a value, blue 
def get_color(value, cap):
    #If the value is below the cap, returns a neutral color.
    if abs(value) <= cap:
        return "#D3D3D3"  # Neutral color for small delays (gray)

    # Color mapping for significant delays
    color_map = {
        -15: "#0D47A1", -10: "#1976D2", -5: "#42A5F5", 
        -3: "#90CAF9", -1: "#E3F2FD", 1: "#FFEBEE", 
        3: "#FFCDD2", 5: "#E57373", 10: "#D32F2F", 
        15: "#B71C1C"
            }
            
    if value < 0:
        for key in sorted(color_map.keys(), reverse=True):
            if value >= key:
                return color_map[key]
    else:
        for key in sorted(color_map.keys()):
            if value <= key:
                return color_map[key]

    return color_map[key]