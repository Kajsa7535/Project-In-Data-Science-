import pandas as pd
from network.model import Network

# Load the smaller network data
file_path = r"data/test_network_week_45_bigger.csv"
#file_path = r"data/smaller_test_network.csv"
#print("staring to load data")
df_network = pd.read_csv(file_path, sep=',', encoding='utf-8')
print("loaded data")
# Initialize the network
network = Network()
network.initate_network(df_network)  # Initialize the network with the data
print("initialized network")
# Number of time steps to visualize
time_steps = 1
print("evaluating network")
network.evaluate_network(df_network, time_steps, visualize=True)



    
