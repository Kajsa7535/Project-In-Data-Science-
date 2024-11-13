
import pandas as pd
import os
from network.model import Network

#smaller network data
#file_path = os.path.join("data", "test_network_week_45_bigger.csv")
#print pwd
print(os.getcwd())
file_path = r"data/test_network_week_45_bigger.csv"
df_network = pd.read_csv(file_path, sep=',', encoding='utf-8')

network = Network()

network.initate_network(df_network)
#network.print_network_info()
print("initial delay matrix")
network.print_delay_matrix(print_all=False)
print(" --------------------------------------------")
print(" ")

network.print_station_info("Karlberg")

network.print_station_info("Stockholm C")
network.print_station_info("Tomteboda övre")

print("edges STHLM -> SOLNA")
network.print_edge_info("Stockholm C", "Karlberg")
network.print_edge_info("Karlberg", "Tomteboda övre")
network.print_edge_info("Tomteboda övre", "Solna")

print("edges SOLNA -> STHLM")
network.print_edge_info("Solna", "Tomteboda övre")
network.print_edge_info("Tomteboda övre", "Karlberg")
network.print_edge_info("Karlberg", "Stockholm C")

time_steps = 5
for i in range(time_steps):
    network.call_time_step()
    network.print_delay_matrix(print_all=False)
    print(" --------------------------------------------")
    print(" ")