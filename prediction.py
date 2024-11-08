
import pandas as pd
from network.model import Network

#smaller network data
file_path = "data/test_network_week_45_bigger.csv"
df_network = pd.read_csv(file_path, sep=',', encoding='utf-8')

network = Network()

network.initate_network(df_network)
#network.print_network_info()
print(" --------------------------------------------")
print(" ")

time_steps = 10
for i in range(time_steps):
    network.call_time_step()
    network.print_delay_matrix()
    print(" --------------------------------------------")
    print(" ")