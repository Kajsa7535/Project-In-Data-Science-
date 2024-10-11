
import pandas as pd
from network.model_new import Network

#smaller network data
file_path = "data/smaller_test_network.csv"
df_network = pd.read_csv(file_path, sep=',', encoding='utf-8')

df_network

network = Network()
network.initate_network(df_network)

#network.print_network_info()

network.call_time_step()
network.call_time_step()

network.print_network_info()
