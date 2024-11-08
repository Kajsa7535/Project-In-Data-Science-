
import pandas as pd
from network.model_new import Network

#smaller network data
file_path = "data/rows_used_for_routes.csv"
df_network = pd.read_csv(file_path, sep=',', encoding='utf-8')

network = Network()

network.initate_network(df_network)
print("Delay matrix at start: ") 
print(network.D_matrix)
print(" --------------------------------------------")
print(" ")

# time_steps = 10
# for i in range(time_steps):
#     network.call_time_step()
#     print("Delay matrix at time step: ", i+1) 
#     print(network.D_matrix)
#     print(" --------------------------------------------")
#     print(" ")